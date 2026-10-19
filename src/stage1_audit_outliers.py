"""
Stage 1: Data Validity Checks, Target Leakage Audit, and Outlier Analysis.

This standalone module executes:
1. Structural & Data Validity Checks:
   - Missing/null values, duplicates, negative numeric bounds, and schema validation.
2. Domain & Target Leakage Audit (via DatasetAdapter):
   - Verifies categorical fraud constraints (0% fraud in PAYMENT, CASH_IN, DEBIT).
   - Evaluates the naive `isFlaggedFraud` rule.
   - Quantifies "The Ledger Bug" target leakage (destination balances failing to update
     during synthetic fraud) and the 100% account depletion trigger.
3. Univariate & Multivariate Outlier Detection:
   - Computes Interquartile Range (IQR) and Modified Z-Score univariate outliers on
     numeric columns (`amount`, `oldbalanceOrg`, `oldbalanceDest`).
   - Runs `IsolationForest` multivariate anomaly detection on a stratified sample of
     `TRANSFER` and `CASH_OUT` transactions to analyze how statistical outliers overlap
     with ground-truth `isFraud`.
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
from sklearn.ensemble import IsolationForest

from src.dataset_adapter import get_dataset_adapter, load_config


def compute_validity_checks(df: pd.DataFrame, numeric_cols: list[str]) -> Dict[str, Any]:
    """
    Verify fundamental data integrity: nulls, exact duplicate rows, infinite values,
    and negative values in financial amount/balance columns.
    """
    null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    total_nulls = sum(null_counts.values())

    negative_counts = {}
    inf_counts = {}
    for col in numeric_cols:
        if col in df.columns:
            vals = df[col].to_numpy()
            negative_counts[col] = int((vals < 0).sum())
            inf_counts[col] = int(np.isinf(vals).sum())

    total_negatives = sum(negative_counts.values())
    total_infs = sum(inf_counts.values())

    # Check exact duplicate rows across key transaction fields
    dup_count = int(df.duplicated().sum())

    return {
        "is_valid_schema": bool(total_nulls == 0 and total_negatives == 0 and total_infs == 0),
        "total_null_values": total_nulls,
        "null_counts_by_column": null_counts,
        "total_negative_values": total_negatives,
        "negative_counts_by_column": negative_counts,
        "total_infinite_values": total_infs,
        "exact_duplicate_rows": dup_count,
    }


def compute_outlier_analysis(df: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    """
    Perform univariate (IQR & Modified Z-score) and multivariate (Isolation Forest)
    outlier detection on the active fraud-capable domain (TRANSFER & CASH_OUT),
    quantifying the overlap between extreme statistical outliers and actual fraud.
    """
    if "type" in df.columns:
        work = df.loc[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    else:
        work = df.copy()

    n_rows = len(work)
    fraud_mask = work["isFraud"].to_numpy() == 1
    n_fraud = int(fraud_mask.sum())

    univariate_stats: Dict[str, Any] = {}
    for col in ["amount", "oldbalanceOrg", "oldbalanceDest"]:
        if col not in work.columns:
            continue
        arr = work[col].to_numpy(dtype=float)
        q25, q50, q75, q99 = np.percentile(arr, [25, 50, 75, 99])
        iqr = q75 - q25
        upper_fence = q75 + 1.5 * iqr
        extreme_fence = q75 + 3.0 * iqr

        is_iqr_outlier = arr > upper_fence
        is_extreme_outlier = arr > extreme_fence

        # Modified Z-score using Median Absolute Deviation (MAD)
        mad = float(np.median(np.abs(arr - q50)))
        denom = 1.4826 * mad if mad > 1e-9 else (float(np.std(arr)) + 1e-9)
        mod_z = np.abs(arr - q50) / denom
        is_modz_outlier = mod_z > 3.5

        fraud_in_iqr = int((is_iqr_outlier & fraud_mask).sum())
        univariate_stats[col] = {
            "median": round(float(q50), 2),
            "q25": round(float(q25), 2),
            "q75": round(float(q75), 2),
            "q99": round(float(q99), 2),
            "max": round(float(arr.max()), 2),
            "iqr": round(float(iqr), 2),
            "iqr_upper_fence": round(float(upper_fence), 2),
            "iqr_outlier_count": int(is_iqr_outlier.sum()),
            "iqr_outlier_pct": round(100.0 * float(is_iqr_outlier.mean()), 2),
            "extreme_3x_iqr_outlier_count": int(is_extreme_outlier.sum()),
            "mod_zscore_gt_3_5_count": int(is_modz_outlier.sum()),
            "fraud_captured_in_iqr_outliers": fraud_in_iqr,
            "pct_of_all_fraud_is_iqr_outlier": round(
                100.0 * fraud_in_iqr / max(n_fraud, 1), 2
            ),
        }

    # Multivariate Outlier Detection via Isolation Forest
    # Sample all fraud cases + 100,000 legitimate TRANSFER/CASH_OUT rows for fast fitting
    pos_df = work.loc[work["isFraud"] == 1]
    neg_df = work.loc[work["isFraud"] == 0]
    neg_sample = neg_df.sample(n=min(100_000, len(neg_df)), random_state=seed)
    iso_df = pd.concat([pos_df, neg_sample], axis=0).reset_index(drop=True)

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
        "analyzed_rows": int(n_rows),
        "univariate_outliers": univariate_stats,
        "multivariate_isolation_forest": multivariate_stats,
    }


def generate_stage1_plots(
    df: pd.DataFrame,
    audit_results: Dict[str, Any],
    figures_dir: Path,
) -> Dict[str, str]:
    """
    Generate diagnostic plots illustrating:
    1. Transaction volume & fraud concentration by type + The Ledger Bug leakage proof.
    2. Outlier distributions (log-amount & account depletion ratio across Legitimate vs Fraud).
    3. Micro-seasonality (Hourly fraud rate vs overall transaction volume).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------------------
    # Figure 1: Categorical Constraint & Target Leakage (The Ledger Bug)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=140)

    type_bd = audit_results["domain_audit"]["type_breakdown"]
    types = list(type_bd.keys())
    fraud_counts = [type_bd[t]["fraud_transactions"] for t in types]
    colors = ["#d95f02" if c > 0 else "#999999" for c in fraud_counts]

    bars = axes[0].bar(types, fraud_counts, color=colors, edgecolor="#222222", alpha=0.9)
    axes[0].set_title("PaySim Categorical Constraint: Fraud Count by Type", fontsize=12, fontweight="bold")
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
    axes[0].set_ylim(0, max(fraud_counts) * 1.18)

    # Panel B: Ledger Bug & Account Depletion Trigger Comparison (Legit vs Fraud)
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
    b1 = axes[1].bar(x_idx - width / 2, legit_pcts, width, label="Legitimate (TRANSFER/CASH_OUT)", color="#377eb8")
    b2 = axes[1].bar(x_idx + width / 2, fraud_pcts, width, label="Fraudulent (isFraud = 1)", color="#e41a1c")
    axes[1].set_title("PaySim Simulation Artifacts & Ledger Bug Prevalence", fontsize=12, fontweight="bold")
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
    tc_df = df.loc[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=140)

    # Panel A: Log10(Amount + 1) distribution for Legit vs Fraud with IQR outlier threshold
    legit_amt = np.log10(tc_df.loc[tc_df["isFraud"] == 0, "amount"].sample(n=min(100_000, len(tc_df)), random_state=42) + 1.0)
    fraud_amt = np.log10(tc_df.loc[tc_df["isFraud"] == 1, "amount"] + 1.0)
    iqr_fence = audit_results["outlier_analysis"]["univariate_outliers"]["amount"]["iqr_upper_fence"]

    axes[0].hist(legit_amt, bins=50, density=True, alpha=0.6, color="#377eb8", label="Legitimate")
    axes[0].hist(fraud_amt, bins=50, density=True, alpha=0.6, color="#e41a1c", label="Fraud (isFraud=1)")
    axes[0].axvline(np.log10(iqr_fence + 1.0), color="#111111", linestyle="--", linewidth=1.8, label=f"IQR Fence ({iqr_fence:,.0f})")
    axes[0].set_title("Transaction Amount Distribution & IQR Outlier Boundary", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("log10(Amount + 1)")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True)

    # Panel B: Micro-seasonality (Hour of Day fraud rate vs transaction volume)
    hours = tc_df["step"] % 24
    hourly_stats = (
        pd.DataFrame({"hour": hours, "isFraud": tc_df["isFraud"]})
        .groupby("hour")
        .agg(total=("isFraud", "count"), fraud=("isFraud", "sum"))
        .reset_index()
    )
    hourly_stats["fraud_rate_pct"] = 100.0 * hourly_stats["fraud"] / hourly_stats["total"].clip(lower=1)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.bar(hourly_stats["hour"], hourly_stats["total"] / 1000.0, color="#b3cde3", alpha=0.7, label="Tx Volume (Thousands)")
    ax2_twin.plot(hourly_stats["hour"], hourly_stats["fraud_rate_pct"], color="#e41a1c", marker="o", linewidth=2.2, label="Fraud Rate (%)")

    ax2.set_title("Micro-Seasonality: Hourly Volume vs. Nocturnal Fraud Spike", fontsize=11.5, fontweight="bold")
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
    """Execute Stage 1 end-to-end and persist audit artifacts."""
    t0 = time.time()
    config = load_config(config_path)
    adapter = get_dataset_adapter(config)

    print("[Stage 1] Loading raw dataset via adapter...")
    df = adapter.load_raw_data()
    print(f"[Stage 1] Loaded {len(df):,} rows and {len(df.columns)} columns.")

    print("[Stage 1] Running data validity checks...")
    validity = compute_validity_checks(df, config["schema"]["numeric_cols"])

    print("[Stage 1] Running domain & target leakage (Ledger Bug) audit...")
    domain_audit = adapter.compute_domain_audit_metrics(df)

    print("[Stage 1] Running univariate (IQR/MAD) & multivariate (IsolationForest) outlier analysis...")
    seed = config.get("remediation", {}).get("random_seed", 42)
    outlier_analysis = compute_outlier_analysis(df, seed=seed)

    audit_results: Dict[str, Any] = {
        "dataset_name": config["dataset_name"],
        "validity_checks": validity,
        "domain_audit": domain_audit,
        "outlier_analysis": outlier_analysis,
    }

    print("[Stage 1] Generating diagnostic figures...")
    figures_dir = Path(config["paths"]["figures_dir"])
    plot_paths = generate_stage1_plots(df, audit_results, figures_dir)
    audit_results["Generated_Plots"] = plot_paths
    audit_results["execution_time_sec"] = round(time.time() - t0, 2)

    out_path = Path(config["paths"]["audit_dir"]) / "audit_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"[Stage 1] Complete in {audit_results['execution_time_sec']}s -> saved {out_path}")
    return audit_results


if __name__ == "__main__":
    run_stage1()
