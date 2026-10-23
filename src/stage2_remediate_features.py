"""
Stage 2: Ledger Remediation, Behavioral Feature Engineering, and Train/Test Splitting.

This standalone module executes:
1. Categorical Segmentation & Subsampling:
   - Restricts modeling data to `TRANSFER` and `CASH_OUT` (where 100% of PaySim fraud occurs).
   - Optionally performs stratified subsampling of negative instances while preserving 100%
     of positive fraud transactions (`isFraud == 1`).
2. Ledger Remediation (Eliminating Target Leakage):
   - Programmatically recalculates and overwrites `newbalanceOrig` and `newbalanceDest`
     using `oldbalance` and `amount` (`newbalanceOrig = max(0, oldbalanceOrg - amount)`,
     `newbalanceDest = oldbalanceDest + amount`) so tree models cannot cheat on the
     simulator's double-entry bookkeeping bug.
3. Behavioral & Micro-Seasonality Feature Engineering:
   - Derives `account_depletion_ratio` (`amount / oldbalanceOrg`), `dest_surge_ratio`,
     zero-balance indicators, and micro-seasonality (`hour_of_day`, `day_of_week`).
4. Standardized Parquet Splits:
   - Saves stratified train and test splits (`X_train.parquet`, `X_test.parquet`,
     `y_train.parquet`, `y_test.parquet`) plus feature metadata (`feature_metadata.json`).
   - Also saves a small leaked-ledger feature split (`X_train_leaked.parquet`,
     `X_test_leaked.parquet`) solely for the Stage 3 ablation experiment proving
     how naive models artificially achieve ~0.999 PR-AUC by exploiting the Ledger Bug.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset_adapter import get_dataset_adapter, load_config


def run_stage2(config_path: str | Path = "configs/paysim_config.yaml") -> Dict[str, Any]:
    """Execute Stage 2 end-to-end and persist Parquet splits + metadata."""
    t0 = time.time()
    config = load_config(config_path)
    adapter = get_dataset_adapter(config)

    print("[Stage 2] Loading raw dataset via adapter...")
    raw_df = adapter.load_raw_data()

    print("[Stage 2] Filtering to valid fraud types (TRANSFER, CASH_OUT) and subsampling...")
    filtered_df = adapter.filter_and_subsample(raw_df)
    del raw_df

    print(
        f"[Stage 2] Filtered modeling dataset: {len(filtered_df):,} rows "
        f"({int(filtered_df[adapter.target_col].sum()):,} fraud cases, "
        f"{100.0 * float(filtered_df[adapter.target_col].mean()):.3f}% positive class)."
    )

    # 1. Prepare Remediated (Fixed Ledger) Feature Matrix
    print("[Stage 2] Fixing ledger balances and engineering behavioral/temporal features...")
    X_clean, y, feature_descriptions = adapter.prepare_modeling_dataframe(
        filtered_df, fix_leakage=True
    )

    # 2. Prepare Leaked (Unfixed Ledger) Feature Matrix for Ablation Comparison
    X_leaked, _, leaked_descriptions = adapter.prepare_modeling_dataframe(
        filtered_df, fix_leakage=False
    )

    test_size = float(config["modeling"].get("test_size", 0.25))
    seed = int(config["modeling"].get("random_seed", 42))

    # Use identical row indices for both clean and leaked splits
    indices = X_clean.index.to_numpy()
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y.to_numpy(),
    )

    X_train = X_clean.iloc[train_idx].reset_index(drop=True)
    X_test = X_clean.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    X_train_leaked = X_leaked.iloc[train_idx].reset_index(drop=True)
    X_test_leaked = X_leaked.iloc[test_idx].reset_index(drop=True)

    modeling_dir = Path(config["paths"]["modeling_dir"])
    modeling_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 2] Saving Parquet splits to {modeling_dir}...")
    X_train.to_parquet(modeling_dir / "X_train.parquet", index=False)
    X_test.to_parquet(modeling_dir / "X_test.parquet", index=False)
    y_train.to_frame(name="isFraud").to_parquet(modeling_dir / "y_train.parquet", index=False)
    y_test.to_frame(name="isFraud").to_parquet(modeling_dir / "y_test.parquet", index=False)

    X_train_leaked.to_parquet(modeling_dir / "X_train_leaked.parquet", index=False)
    X_test_leaked.to_parquet(modeling_dir / "X_test_leaked.parquet", index=False)

    metadata: Dict[str, Any] = {
        "dataset_name": config["dataset_name"],
        "fix_ledger_bug_applied": True,
        "valid_fraud_types_retained": config["remediation"].get("valid_fraud_types"),
        "total_modeling_rows": int(len(X_clean)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_fraud_count": int(y_train.sum()),
        "test_fraud_count": int(y_test.sum()),
        "train_fraud_rate_pct": round(100.0 * float(y_train.mean()), 4),
        "test_fraud_rate_pct": round(100.0 * float(y_test.mean()), 4),
        "feature_names": list(X_clean.columns),
        "feature_descriptions": feature_descriptions,
        "leaked_feature_names": list(X_leaked.columns),
        "execution_time_sec": round(time.time() - t0, 2),
    }

    meta_path = modeling_dir / "feature_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Stage 2] Complete in {metadata['execution_time_sec']}s -> saved {meta_path}")
    return metadata


if __name__ == "__main__":
    run_stage2()
