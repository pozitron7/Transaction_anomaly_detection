"""
Stage 1: Data Validity Checks, Target Leakage Audit, and Outlier Analysis (Apache Spark).

This standalone module executes:
1. Structural & Data Validity Checks (Spark distributed single-pass):
   - Missing/null/NaN values, duplicates, negative numeric bounds, infinite values.
2. Domain & Target Leakage Audit (via Spark DatasetAdapter):
   - Verifies categorical fraud constraints (0% fraud in PAYMENT, CASH_IN, DEBIT).
   - Evaluates the naive `isFlaggedFraud` rule.
   - Quantifies "The Ledger Bug" target leakage and the 100% account depletion trigger.
3. Univariate & Multivariate Outlier Detection:
   - Computes distributed approximate quantiles (IQR) and Modified Z-Score (MAD)
     univariate outliers on `amount`, `oldbalanceOrg`, `oldbalanceDest`.
   - Runs `IsolationForest` multivariate anomaly detection on a bounded Spark-sampled
     subset of `TRANSFER` and `CASH_OUT` transactions.
4. Diagnostic Visualizations:
   - Saves publication-quality charts to `reports/figures/` and structured JSON metrics
     to `data/01_audit/audit_summary.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType
from sklearn.ensemble import IsolationForest

from src.dataset_adapter import get_dataset_adapter, load_config
from src.spark_utils import get_spark_session


def compute_validity_checks(df: DataFrame, numeric_cols: list[str]) -> Dict[str, Any]:
    """
    Verify fundamental data integrity in Spark: nulls/NaNs, exact duplicate rows,
    infinite values, and negative values in financial amount/balance columns.
    """
    schema_map = {field.name: field.dataType for field in df.schema.fields}
    agg_exprs = [F.count("*").alias("__total_rows")]

    for col in df.columns:
        is_float = isinstance(schema_map.get(col), (DoubleType, FloatType))
        cond = F.col(col).isNull() | F.isnan(F.col(col)) if is_float else F.col(col).isNull()
        agg_exprs.append(F.sum(F.when(cond, 1).otherwise(0)).alias(f"__null_{col}"))

    present_numeric = [col for col in numeric_cols if col in df.columns]
    for col in present_numeric:
        agg_exprs.append(
            F.sum(F.when(F.col(col) < 0, 1).otherwise(0)).alias(f"__neg_{col}")
        )
        agg_exprs.append(
            F.sum(
                F.when(
                    (F.col(col) == float("inf")) | (F.col(col) == float("-inf")), 1
                ).otherwise(0)
            ).alias(f"__inf_{col}")
        )

    row = df.agg(*agg_exprs).collect()[0]
    total_rows = int(row["__total_rows"] or 0)

    null_counts = {col: int(row[f"__null_{col}"] or 0) for col in df.columns}
    negative_counts = {col: int(row[f"__neg_{col}"] or 0) for col in present_numeric}
    inf_counts = {col: int(row[f"__inf_{col}"] or 0) for col in present_numeric}

    total_nulls = sum(null_counts.values())
    total_negatives = sum(negative_counts.values())
    total_infs = sum(inf_counts.values())

    # Distributed duplicate detection in Spark
    dup_count = int(total_rows - df.dropDuplicates().count())

    return {
        "is_valid_schema": bool(total_nulls == 0 and total_negatives == 0 and total_infs == 0),
        "total_null_values": total_nulls,
        "null_counts_by_column": null_counts,
        "total_negative_values": total_negatives,
        "negative_counts_by_column": negative_counts,
        "total_infinite_values": total_infs,
        "exact_duplicate_rows": dup_count,
    }


def compute_outlier_analysis(df: DataFrame, seed: int = 42) -> Dict[str, Any]:
    """
    Perform univariate (distributed IQR & Modified Z-score via Spark approxQuantile)
    and multivariate (Isolation Forest on a bounded Spark sample) outlier detection
    on the active fraud-capable domain (TRANSFER & CASH_OUT).
    """
    work = df.filter(F.col("type").isin(["TRANSFER", "CASH_OUT"])) if "type" in df.columns else df

    counts_row = work.agg(
        F.count("*").alias("n_rows"),
        F.sum(F.when(F.col("isFraud") == 1, 1).otherwise(0)).alias("n_fraud"),
    ).collect()[0]
    n_rows = int(counts_row["n_rows"] or 0)
    n_fraud = int(counts_row["n_fraud"] or 0)

    univariate_stats: Dict[str, Any] = {}
    for col in ["amount", "oldbalanceOrg", "oldbalanceDest"]:
        if col not in work.columns:
            continue

        col_df = work.select(F.col(col).cast("double").alias(col), "isFraud")
        q25, q50, q75, q99 = col_df.approxQuantile(col, [0.25, 0.50, 0.75, 0.99], 0.001)
        iqr = q75 - q25
        upper_fence = q75 + 1.5 * iqr
        extreme_fence = q75 + 3.0 * iqr

        # Modified Z-score using distributed Median Absolute Deviation (MAD)
        mad = col_df.select(F.abs(F.col(col) - F.lit(q50)).alias("abs_dev")).approxQuantile(
            "abs_dev", [0.5], 0.001
        )[0]
        if mad > 1e-9:
            denom = 1.4826 * mad
        else:
            std_val = col_df.agg(F.stddev(col)).collect()[0][0]
            denom = float(std_val or 0.0) + 1e-9

        is_iqr_outlier = F.col(col) > F.lit(upper_fence)
        is_extreme_outlier = F.col(col) > F.lit(extreme_fence)
        is_modz_outlier = (F.abs(F.col(col) - F.lit(q50)) / F.lit(denom)) > F.lit(3.5)

        stats_row = col_df.agg(
            F.max(col).alias("max_val"),
            F.sum(F.when(is_iqr_outlier, 1).otherwise(0)).alias("iqr_cnt"),
            F.sum(F.when(is_extreme_outlier, 1).otherwise(0)).alias("ext_cnt"),
            F.sum(F.when(is_modz_outlier, 1).otherwise(0)).alias("modz_cnt"),
            F.sum(F.when(is_iqr_outlier & (F.col("isFraud") == 1), 1).otherwise(0)).alias(
                "fraud_iqr_cnt"
            ),
        ).collect()[0]

        iqr_cnt = int(stats_row["iqr_cnt"] or 0)
        fraud_in_iqr = int(stats_row["fraud_iqr_cnt"] or 0)

        univariate_stats[col] = {
            "median": round(float(q50), 2),
            "q25": round(float(q25), 2),
            "q75": round(float(q75), 2),
            "q99": round(float(q99), 2),
            "max": round(float(stats_row["max_val"] or 0.0), 2),
            "iqr": round(float(iqr), 2),
            "iqr_upper_fence": round(float(upper_fence), 2),
            "iqr_outlier_count": iqr_cnt,
            "iqr_outlier_pct": round(100.0 * iqr_cnt / max(n_rows, 1), 2),
            "extreme_3x_iqr_outlier_count": int(stats_row["ext_cnt"] or 0),
            "mod_zscore_gt_3_5_count": int(stats_row["modz_cnt"] or 0),
            "fraud_captured_in_iqr_outliers": fraud_in_iqr,
            "pct_of_all_fraud_is_iqr_outlier": round(100.0 * fraud_in_iqr / max(n_fraud, 1), 2),
        }

    # Multivariate Outlier Detection via Isolation Forest on a bounded Spark sample
    iso_cols = ["amount", "oldbalanceOrg", "oldbalanceDest", "isFraud"]
    pos_spark = work.filter(F.col("isFraud") == 1).select(*iso_cols)
    if n_fraud > 100_000:
        pos_spark = pos_spark.sample(False, min(1.0, 105_000.0 / n_fraud), seed).limit(100_000)

    n_legit = max(n_rows - n_fraud, 1)
    neg_target = min(100_000, n_legit)
    neg_frac = min(1.0, (neg_target * 1.05) / float(n_legit))
    neg_spark = (
        work.filter(F.col("isFraud") == 0)
        .select(*iso_cols)
        .sample(withReplacement=False, fraction=neg_frac, seed=seed)
        .limit(neg_target)
    )

    iso_df = pos_spark.unionByName(neg_spark).toPandas()

    iso_features = ["amount", "oldbalanceOrg", "oldbalanceDest"]
    X_iso = np.log1p(np.maximum(0.0, iso_df[iso_features].to_numpy(dtype=float)))
    y_iso = iso_df["isFraud"].to_numpy(dtype=int)

    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=seed,
        n_jobs=-1,
    )
    preds = iso_model.fit_predict(X_iso)  # -1 = outlier, 1 = inlier
    scores = -iso_model.score_samples(X_iso)  # Higher = more anomalous

    is_multi_outlier = preds == -1
    fraud_flagged_by_iso = int((is_multi_outlier & (y_iso == 1)).sum())
    total_fraud_iso = int((y_iso == 1).sum())

    multivariate_stats = {
        "evaluated_sample_size": int(len(iso_df)),
        "contamination_rate": 0.05,
        "total_outliers_flagged": int(is_multi_outlier.sum()),
        "fraud_flagged_as_multivariate_outlier": fraud_flagged_by_iso,
        "pct_of_fraud_flagged_as_multivariate_outlier": round(
            100.0 * fraud_flagged_by_iso / max(total_fraud_iso, 1), 2
        ),
        "mean_anomaly_score_fraud": round(float(scores[y_iso == 1].mean()), 4),
        "mean_anomaly_score_legit": round(float(scores[y_iso == 0].mean()), 4),
    }

    return {
        "analyzed_subset": "TRANSFER and CASH_OUT",
        "analyzed_rows": n_rows,
        "univariate_outliers": univariate_stats,
        "multivariate_isolation_forest": multivariate_stats,
    }


def generate_stage1_plots(
    df: DataFrame,
    audit_results: Dict[str, Any],
    figures_dir: Path,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Generate diagnostic plots using Spark-aggregated statistics and bounded samples.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )

    # -------------------------------------------------------------------------
    # Figure 1: Categorical Constraint & Target Leakage (The Ledger Bug)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=140)

    type_bd = audit_results["domain_audit"]["type_breakdown"]
    types = list(type_bd.keys())
    fraud_counts = [type_bd[t]["fraud_transactions"] for t in types]
    colors = ["#d95f02" if c > 0 else "#999999" for c in fraud_counts]

    bars = axes[0].bar(types, fraud_counts, color=colors, edgecolor="#222222", alpha=0.9)
    axes[0].set_title(
        "PaySim Categorical Constraint: Fraud Count by Type", fontsize=12, fontweight="bold"
    )
    axes[0].set_ylabel("Number of Fraudulent Transactions")
    axes[0].set_xlabel("Transaction Type")
    for bar, cnt in zip(bars, fraud_counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 80,
            f"{cnt:,}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(fraud_counts) * 1.18 if fraud_counts else 1)

    lb = audit_results["domain_audit"]["ledger_bug_and_leakage_audit"]
    categories = [
        "Dest Balance Unchanged\n(newDest == oldDest)",
        "Dest Balance Remains 0\n(oldDest=0 & newDest=0)",
        "100% Account Drained\n(amount == oldbalanceOrg)",
    ]
    fraud_pcts = [
        lb["dest_balance_unchanged_in_fraud_pct"],
        lb["dest_zero_before_and_after_in_fraud_pct"],
        lb["exact_100pct_account_depletion_in_fraud_pct"],
    ]
    legit_pcts = [
        lb["dest_balance_unchanged_in_legit_pct"],
        lb["dest_zero_before_and_after_in_legit_pct"],
        lb["exact_100pct_account_depletion_in_legit_pct"],
    ]

    x_idx = np.arange(len(categories))
    width = 0.36
    b1 = axes[1].bar(
        x_idx - width / 2, legit_pcts, width, label="Legitimate (TRANSFER/CASH_OUT)", color="#377eb8"
    )
    b2 = axes[1].bar(
        x_idx + width / 2, fraud_pcts, width, label="Fraudulent (isFraud = 1)", color="#e41a1c"
    )
    axes[1].set_title(
        "PaySim Simulation Artifacts & Ledger Bug Prevalence", fontsize=12, fontweight="bold"
    )
    axes[1].set_ylabel("Prevalence within Class (%)")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(categories, fontsize=9)
    axes[1].set_ylim(0, 115)
    axes[1].legend(loc="upper left", frameon=True)

    for b in list(b1) + list(b2):
        h = b.get_height()
        axes[1].text(
            b.get_x() + b.get_width() / 2.0,
            h + 1.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    plt.tight_layout()
    fig1_path = figures_dir / "stage1_leakage_and_types.png"
    fig.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Outliers & Micro-Seasonality (Hourly Fraud Pattern)
    # -------------------------------------------------------------------------
    tc_df = df.filter(F.col("type").isin(["TRANSFER", "CASH_OUT"]))
    tc_rows = max(audit_results["outlier_analysis"]["analyzed_rows"], 1)
    legit_frac = min(1.0, 105_000.0 / float(tc_rows))

    legit_amt_pd = (
        tc_df.filter(F.col("isFraud") == 0)
        .select("amount")
        .sample(withReplacement=False, fraction=legit_frac, seed=seed)
        .limit(100_000)
        .toPandas()
    )
    fraud_amt_pd = tc_df.filter(F.col("isFraud") == 1).select("amount").limit(100_000).toPandas()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=140)
    legit_amt = np.log10(legit_amt_pd["amount"].to_numpy(dtype=float) + 1.0)
    fraud_amt = np.log10(fraud_amt_pd["amount"].to_numpy(dtype=float) + 1.0)
    iqr_fence = audit_results["outlier_analysis"]["univariate_outliers"]["amount"]["iqr_upper_fence"]

    axes[0].hist(legit_amt, bins=50, density=True, alpha=0.6, color="#377eb8", label="Legitimate")
    axes[0].hist(fraud_amt, bins=50, density=True, alpha=0.6, color="#e41a1c", label="Fraud (isFraud=1)")
    axes[0].axvline(
        np.log10(iqr_fence + 1.0),
        color="#111111",
        linestyle="--",
        linewidth=1.8,
        label=f"IQR Fence ({iqr_fence:,.0f})",
    )
    axes[0].set_title(
        "Transaction Amount Distribution & IQR Outlier Boundary", fontsize=11.5, fontweight="bold"
    )
    axes[0].set_xlabel("log10(Amount + 1)")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True)

    # Panel B: Micro-seasonality aggregated in Spark (only 24 rows collected)
    hourly_stats = (
        tc_df.groupBy((F.col("step") % F.lit(24)).alias("hour"))
        .agg(
            F.count("*").alias("total"),
            F.sum(F.col("isFraud").cast("long")).alias("fraud"),
        )
        .orderBy("hour")
        .toPandas()
    )
    hourly_stats["fraud_rate_pct"] = (
        100.0 * hourly_stats["fraud"] / hourly_stats["total"].clip(lower=1)
    )

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.bar(
        hourly_stats["hour"],
        hourly_stats["total"] / 1000.0,
        color="#b3cde3",
        alpha=0.7,
        label="Tx Volume (Thousands)",
    )
    ax2_twin.plot(
        hourly_stats["hour"],
        hourly_stats["fraud_rate_pct"],
        color="#e41a1c",
        marker="o",
        linewidth=2.2,
        label="Fraud Rate (%)",
    )

    ax2.set_title(
        "Micro-Seasonality: Hourly Volume vs. Nocturnal Fraud Spike", fontsize=11.5, fontweight="bold"
    )
    ax2.set_xlabel("Hour of Day (step mod 24)")
    ax2.set_ylabel("Transaction Volume (Thousands)", color="#2b5c8f")
    ax2_twin.set_ylabel("Fraud Rate (%)", color="#e41a1c")
    ax2.set_xticks(range(0, 24, 2))
    ax2_twin.grid(False)

    plt.tight_layout()
    fig2_path = figures_dir / "stage1_outliers_and_seasonality.png"
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "leakage_and_types_plot": str(fig1_path),
        "outliers_and_seasonality_plot": str(fig2_path),
    }


def run_stage1(config_path: str | Path = "configs/paysim_config.yaml") -> Dict[str, Any]:
    """Execute Stage 1 end-to-end using Apache Spark and persist audit artifacts."""
    t0 = time.time()
    config = load_config(config_path)
    adapter = get_dataset_adapter(config)
    spark = get_spark_session(config)

    try:
        print("[Stage 1] Loading raw dataset via Spark adapter...")
        df = adapter.load_raw_data(spark).cache()
        row_count = df.count()
        print(f"[Stage 1] Loaded {row_count:,} rows and {len(df.columns)} columns in Spark.")

        print("[Stage 1] Running distributed data validity checks...")
        validity = compute_validity_checks(df, config["schema"]["numeric_cols"])

        print("[Stage 1] Running distributed domain & target leakage (Ledger Bug) audit...")
        domain_audit = adapter.compute_domain_audit_metrics(df)

        print(
            "[Stage 1] Running distributed univariate (IQR/MAD) & multivariate (IsolationForest) outlier analysis..."
        )
        seed = int(config.get("remediation", {}).get("random_seed", 42))
        outlier_analysis = compute_outlier_analysis(df, seed=seed)

        audit_results: Dict[str, Any] = {
            "dataset_name": config["dataset_name"],
            "engine": f"Apache Spark {spark.version}",
            "validity_checks": validity,
            "domain_audit": domain_audit,
            "outlier_analysis": outlier_analysis,
        }

        print("[Stage 1] Generating diagnostic figures...")
        figures_dir = Path(config["paths"]["figures_dir"])
        plot_paths = generate_stage1_plots(df, audit_results, figures_dir, seed=seed)
        audit_results["Generated_Plots"] = plot_paths
        audit_results["execution_time_sec"] = round(time.time() - t0, 2)

        df.unpersist()
    finally:
        spark.stop()

    out_path = Path(config["paths"]["audit_dir"]) / "audit_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"[Stage 1] Complete in {audit_results['execution_time_sec']}s -> saved {out_path}")
    return audit_results


if __name__ == "__main__":
    run_stage1()
