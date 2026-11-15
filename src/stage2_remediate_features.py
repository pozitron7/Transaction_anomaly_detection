"""
Stage 2: Ledger Remediation, Behavioral Feature Engineering, and Train/Test Splitting (Apache Spark).

This standalone module executes all data preparation in Apache Spark:
1. Categorical Segmentation & Stratified Subsampling:
   - Restricts modeling data to `TRANSFER` and `CASH_OUT` (where 100% of PaySim fraud occurs).
   - Performs distributed stratified subsampling of negative instances while preserving
     100% of positive fraud transactions (`isFraud == 1`).
2. Ledger Remediation (Eliminating Target Leakage):
   - Programmatically recalculates and overwrites `newbalanceOrig` and `newbalanceDest`
     using `oldbalance` and `amount` (`newbalanceOrig = max(0, oldbalanceOrg - amount)`,
     `newbalanceDest = oldbalanceDest + amount`) so tree models cannot cheat on the
     simulator's double-entry bookkeeping bug.
3. Behavioral & Micro-Seasonality Feature Engineering:
   - Derives `account_depletion_ratio`, `dest_surge_ratio`, zero-balance indicators,
     and micro-seasonality (`hour_of_day`, `day_of_week`).
4. Distributed Stratified Train/Test Split & Parquet Export:
   - Saves stratified train and test splits (`X_train.parquet`, `X_test.parquet`,
     `y_train.parquet`, `y_test.parquet`) plus feature metadata (`feature_metadata.json`).
   - Also saves the leaked-ledger feature split (`X_train_leaked.parquet`,
     `X_test_leaked.parquet`) for the Stage 3 ablation experiment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import functions as F

from src.dataset_adapter import get_dataset_adapter, load_config
from src.spark_utils import get_spark_session, write_spark_parquet


def run_stage2(config_path: str | Path = "configs/paysim_config.yaml") -> Dict[str, Any]:
    """Execute Stage 2 end-to-end in Apache Spark and persist Parquet splits + metadata."""
    t0 = time.time()
    config = load_config(config_path)
    adapter = get_dataset_adapter(config)
    spark = get_spark_session(config)
    target_col = adapter.target_col

    modeling_dir = Path(config["paths"]["modeling_dir"])
    modeling_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("[Stage 2] Loading raw dataset via Spark adapter...")
        raw_df = adapter.load_raw_data(spark)

        print("[Stage 2] Filtering to valid fraud types (TRANSFER, CASH_OUT) and subsampling...")
        filtered_df = adapter.filter_and_subsample(raw_df)

        test_size = float(config["modeling"].get("test_size", 0.25))
        seed = int(config["modeling"].get("random_seed", 42))

        # Assign a stratified deterministic split flag once so clean and leaked splits
        # contain the exact same transactions in train and test.
        pos_split = filtered_df.filter(F.col(target_col) == 1).withColumn(
            "_is_test", F.when(F.rand(seed) < test_size, 1).otherwise(0)
        )
        neg_split = filtered_df.filter(F.col(target_col) == 0).withColumn(
            "_is_test", F.when(F.rand(seed + 1) < test_size, 1).otherwise(0)
        )
        split_df = pos_split.unionByName(neg_split).cache()

        summary_row = split_df.agg(
            F.count("*").alias("total_rows"),
            F.sum(F.col(target_col).cast("long")).alias("total_fraud"),
        ).collect()[0]
        total_rows = int(summary_row["total_rows"] or 0)
        total_fraud = int(summary_row["total_fraud"] or 0)

        print(
            f"[Stage 2] Filtered modeling dataset: {total_rows:,} rows "
            f"({total_fraud:,} fraud cases, "
            f"{100.0 * total_fraud / max(total_rows, 1):.3f}% positive class)."
        )

        # 1. Prepare Remediated (Fixed Ledger) Feature Matrix
        print("[Stage 2] Fixing ledger balances and engineering behavioral/temporal features in Spark...")
        clean_df, feature_cols, feature_descriptions = adapter.prepare_modeling_dataframe(
            split_df, fix_leakage=True
        )

        # 2. Prepare Leaked (Unfixed Ledger) Feature Matrix for Ablation Comparison
        leaked_df, leaked_cols, _ = adapter.prepare_modeling_dataframe(
            split_df, fix_leakage=False
        )

        train_clean = clean_df.filter(F.col("_is_test") == 0).cache()
        test_clean = clean_df.filter(F.col("_is_test") == 1).cache()

        train_stats = train_clean.agg(
            F.count("*").alias("rows"),
            F.sum(F.col(target_col).cast("long")).alias("fraud"),
        ).collect()[0]
        test_stats = test_clean.agg(
            F.count("*").alias("rows"),
            F.sum(F.col(target_col).cast("long")).alias("fraud"),
        ).collect()[0]

        train_rows = int(train_stats["rows"] or 0)
        train_fraud = int(train_stats["fraud"] or 0)
        test_rows = int(test_stats["rows"] or 0)
        test_fraud = int(test_stats["fraud"] or 0)

        print(f"[Stage 2] Saving Parquet splits to {modeling_dir}...")
        write_spark_parquet(train_clean.select(*feature_cols), modeling_dir / "X_train.parquet")
        write_spark_parquet(test_clean.select(*feature_cols), modeling_dir / "X_test.parquet")
        write_spark_parquet(
            train_clean.select(F.col(target_col).alias("isFraud")),
            modeling_dir / "y_train.parquet",
        )
        write_spark_parquet(
            test_clean.select(F.col(target_col).alias("isFraud")),
            modeling_dir / "y_test.parquet",
        )

        write_spark_parquet(
            leaked_df.filter(F.col("_is_test") == 0).select(*leaked_cols),
            modeling_dir / "X_train_leaked.parquet",
        )
        write_spark_parquet(
            leaked_df.filter(F.col("_is_test") == 1).select(*leaked_cols),
            modeling_dir / "X_test_leaked.parquet",
        )

        train_clean.unpersist()
        test_clean.unpersist()
        split_df.unpersist()

        spark_version = spark.version
    finally:
        spark.stop()

    metadata: Dict[str, Any] = {
        "dataset_name": config["dataset_name"],
        "engine": f"Apache Spark {spark_version}",
        "fix_ledger_bug_applied": True,
        "valid_fraud_types_retained": config["remediation"].get("valid_fraud_types"),
        "total_modeling_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_fraud_count": train_fraud,
        "test_fraud_count": test_fraud,
        "train_fraud_rate_pct": round(100.0 * train_fraud / max(train_rows, 1), 4),
        "test_fraud_rate_pct": round(100.0 * test_fraud / max(test_rows, 1), 4),
        "feature_names": feature_cols,
        "feature_descriptions": feature_descriptions,
        "leaked_feature_names": leaked_cols,
        "execution_time_sec": round(time.time() - t0, 2),
    }

    meta_path = modeling_dir / "feature_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Stage 2] Complete in {metadata['execution_time_sec']}s -> saved {meta_path}")
    return metadata


if __name__ == "__main__":
    run_stage2()
