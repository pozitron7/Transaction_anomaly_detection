"""
Dataset Adapter & Configuration Layer.

This module provides a clean abstraction between dataset-specific schemas/quirks
(such as the PaySim synthetic dataset and its known Ledger Bug) and downstream
modeling, explainability, and reporting stages.

To switch to a different dataset in the future:
1. Create a new YAML configuration file under `configs/`.
2. Either use `GenericDatasetAdapter` or subclass `BaseDatasetAdapter` to implement
   dataset-specific domain rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str | Path = "configs/paysim_config.yaml") -> Dict[str, Any]:
    """Load and validate the YAML pipeline configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure output directories exist
    for dir_key, dir_val in config.get("paths", {}).items():
        Path(dir_val).mkdir(parents=True, exist_ok=True)

    return config


class BaseDatasetAdapter(ABC):
    """
    Abstract base class defining the contract for any fraud detection dataset.
    Downstream stages (Training, Custom SHAP, Library SHAP, Reporting) interact
    only with standardized feature matrices produced by an adapter.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.schema = config["schema"]
        self.remediation = config.get("remediation", {})

    @property
    def target_col(self) -> str:
        return self.schema["target_col"]

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        """Load the raw dataset from disk using optimized dtypes."""
        pass

    @abstractmethod
    def compute_domain_audit_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute dataset-specific validity checks and artifact/leakage audits.
        Returns a JSON-serializable dictionary of findings.
        """
        pass

    @abstractmethod
    def prepare_modeling_dataframe(
        self, df: pd.DataFrame, fix_leakage: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
        """
        Filter, remediate target leakage, and engineer behavioral features.

        Args:
            df: Raw or filtered DataFrame.
            fix_leakage: If True, fixes known target leakage artifacts (e.g., PaySim Ledger Bug).
                         If False, retains raw leaked features for ablation benchmarking.

        Returns:
            X: Feature DataFrame (all numeric, ready for tree models and SHAP).
            y: Binary target Series (0 = legitimate, 1 = fraud).
            feature_descriptions: Mapping from column name to human-readable description.
        """
        pass


class PaySimAdapter(BaseDatasetAdapter):
    """
    Adapter for the PaySim Synthetic Mobile Money Dataset.

    Explicitly accounts for PaySim's architectural properties and limitations:
    1. Categorical Constraint: Fraud only exists in TRANSFER and CASH_OUT.
    2. Target Leakage (The Ledger Bug): Fraudulent transactions often leave
       destination balances at 0.0 or unchanged. We audit this correlation in Stage 1
       and programmatically recalculate `newbalanceOrig` and `newbalanceDest` in Stage 2.
    3. Behavioral Ratios & Micro-Seasonality: Derives account depletion ratios
       and hour-of-day/day-of-week features over the 744-hour (31-day) simulation.
    """

    def load_raw_data(self) -> pd.DataFrame:
        csv_path = Path(self.config["raw_csv_path"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Raw PaySim CSV not found at {csv_path.resolve()}")

        # Optimized dtypes to load 6.36M rows in ~4-6 seconds with low RAM footprint
        dtype_map = {
            "step": "int32",
            "type": "category",
            "amount": "float64",
            "nameOrig": "string",
            "oldbalanceOrg": "float64",
            "newbalanceOrig": "float64",
            "nameDest": "string",
            "oldbalanceDest": "float64",
            "newbalanceDest": "float64",
            "isFraud": "int8",
            "isFlaggedFraud": "int8",
        }
        df = pd.read_csv(csv_path, dtype=dtype_map)
        return df

    def compute_domain_audit_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Quantify PaySim's structural properties, naive rule performance, entity prefixes,
        and the exact severity of the Ledger Bug target leakage.
        """
        total_rows = int(len(df))
        total_fraud = int(df["isFraud"].sum())

        # 1. Categorical constraint breakdown across all 5 transaction types
        type_Next = (
            df.groupby("type", observed=False)
            .agg(
                total_tx=("isFraud", "count"),
                fraud_tx=("isFraud", "sum"),
                mean_amount=("amount", "mean"),
            )
            .reset_index()
        )
        type_breakdown = {}
        for _, row in type_Next.iterrows():
            t_name = str(row["type"])
            cnt = int(row["total_tx"])
            f_cnt = int(row["fraud_tx"])
            type_breakdown[t_name] = {
                "total_transactions": cnt,
                "share_of_dataset_pct": round(100.0 * cnt / total_rows, 3),
                "fraud_transactions": f_cnt,
                "fraud_rate_pct": round(100.0 * f_cnt / max(cnt, 1), 4),
                "mean_amount": round(float(row["mean_amount"]), 2),
            }

        # 2. Naive business rule (`isFlaggedFraud`) evaluation
        flagged_total = int(df["isFlaggedFraud"].sum())
        flagged_true_pos = int(((df["isFlaggedFraud"] == 1) & (df["isFraud"] == 1)).sum())
        flagged_false_pos = int(((df["isFlaggedFraud"] == 1) & (df["isFraud"] == 0)).sum())
        flagged_false_neg = total_fraud - flagged_true_pos

        # 3. Entity Prefix ('C' vs 'M') topology check
        orig_merchants = int(df["nameOrig"].str.startswith("M").sum())
        dest_merchants = int(df["nameDest"].str.startswith("M").sum())
        dest_merchants_in_transfer_cashout = int(
            df.loc[df["type"].isin(["TRANSFER", "CASH_OUT"]), "nameDest"]
            .str.startswith("M")
            .sum()
        )

        # 4. Quantifying "The Ledger Bug" on TRANSFER and CASH_OUT transactions
        # In TRANSFER and CASH_OUT, money leaves Origin and enters Destination (for TRANSFER)
        # or is cashed out via a Customer/Mule account.
        tc_mask = df["type"].isin(["TRANSFER", "CASH_OUT"])
        tc_df = df.loc[tc_mask]

        # Discrepancy equations:
        # Expected newbalanceOrig = max(0, oldbalanceOrg - amount)
        # Raw ledger error Orig = (oldbalanceOrg - amount) - newbalanceOrig
        # Raw ledger error Dest = (oldbalanceDest + amount) - newbalanceDest
        err_orig = (tc_df["oldbalanceOrg"] - tc_df["amount"]) - tc_df["newbalanceOrig"]
        err_dest = (tc_df["oldbalanceDest"] + tc_df["amount"]) - tc_df["newbalanceDest"]

        fraud_mask = tc_df["isFraud"] == 1
        legit_mask = tc_df["isFraud"] == 0

        # Specifically check how often fraudulent transactions fail to update newbalanceDest
        # (e.g., oldbalanceDest == 0 and newbalanceDest == 0 despite amount > 0, or newbalanceDest == oldbalanceDest)
        dest_unchanged_fraud = float(
            (
                (tc_df.loc[fraud_mask, "newbalanceDest"] == tc_df.loc[fraud_mask, "oldbalanceDest"])
                & (tc_df.loc[fraud_mask, "amount"] > 0)
            ).mean()
            * 100.0
        )
        dest_unchanged_legit = float(
            (
                (tc_df.loc[legit_mask, "newbalanceDest"] == tc_df.loc[legit_mask, "oldbalanceDest"])
                & (tc_df.loc[legit_mask, "amount"] > 0)
            ).mean()
            * 100.0
        )

        dest_zero_after_tx_fraud = float(
            (
                (tc_df.loc[fraud_mask, "oldbalanceDest"] == 0)
                & (tc_df.loc[fraud_mask, "newbalanceDest"] == 0)
                & (tc_df.loc[fraud_mask, "amount"] > 0)
            ).mean()
            * 100.0
        )
        dest_zero_after_tx_legit = float(
            (
                (tc_df.loc[legit_mask, "oldbalanceDest"] == 0)
                & (tc_df.loc[legit_mask, "newbalanceDest"] == 0)
                & (tc_df.loc[legit_mask, "amount"] > 0)
            ).mean()
            * 100.0
        )

        # Account Depletion Trigger (`amount == oldbalanceOrg` & `amount > 0`)
        exact_depletion_fraud = float(
            (
                np.isclose(tc_df.loc[fraud_mask, "amount"], tc_df.loc[fraud_mask, "oldbalanceOrg"])
                & (tc_df.loc[fraud_mask, "amount"] > 0)
            ).mean()
            * 100.0
        )
        exact_depletion_legit = float(
            (
                np.isclose(tc_df.loc[legit_mask, "amount"], tc_df.loc[legit_mask, "oldbalanceOrg"])
                & (tc_df.loc[legit_mask, "amount"] > 0)
            ).mean()
            * 100.0
        )

        # Point-biserial / Pearson correlation with isFraud on TRANSFER/CASH_OUT
        y_tc = tc_df["isFraud"].astype(float)
        corr_err_orig = float(np.corrcoef(np.abs(err_orig), y_tc)[0, 1])
        corr_err_dest = float(np.corrcoef(np.abs(err_dest), y_tc)[0, 1])
        corr_dest_unchanged = float(
            np.corrcoef(
                (tc_df["newbalanceDest"] == tc_df["oldbalanceDest"]).astype(float), y_tc
            )[0, 1]
        )

        return {
            "total_rows": total_rows,
            "total_fraud": total_fraud,
            "global_fraud_rate_pct": round(100.0 * total_fraud / total_rows, 4),
            "step_min": int(df["step"].min()),
            "step_max": int(df["step"].max()),
            "type_breakdown": type_breakdown,
            "naive_isFlaggedFraud_audit": {
                "total_flagged": flagged_total,
                "true_positives": flagged_true_pos,
                "false_positives": flagged_false_pos,
                "false_negatives": flagged_false_neg,
                "recall_pct": round(100.0 * flagged_true_pos / max(total_fraud, 1), 4),
            },
            "entity_prefix_audit": {
                "orig_merchant_count": orig_merchants,
                "dest_merchant_count": dest_merchants,
                "dest_merchants_in_transfer_or_cashout": dest_merchants_in_transfer_cashout,
            },
            "ledger_bug_and_leakage_audit": {
                "transfer_cashout_rows": int(len(tc_df)),
                "dest_balance_unchanged_in_fraud_pct": round(dest_unchanged_fraud, 2),
                "dest_balance_unchanged_in_legit_pct": round(dest_unchanged_legit, 2),
                "dest_zero_before_and_after_in_fraud_pct": round(dest_zero_after_tx_fraud, 2),
                "dest_zero_before_and_after_in_legit_pct": round(dest_zero_after_tx_legit, 2),
                "exact_100pct_account_depletion_in_fraud_pct": round(exact_depletion_fraud, 2),
                "exact_100pct_account_depletion_in_legit_pct": round(exact_depletion_legit, 2),
                "corr_abs_error_orig_with_fraud": round(corr_err_orig, 4),
                "corr_abs_error_dest_with_fraud": round(corr_err_dest, 4),
                "corr_dest_unchanged_with_fraud": round(corr_dest_unchanged, 4),
                "mean_abs_error_dest_fraud": round(float(np.abs(err_dest[fraud_mask]).mean()), 2),
                "mean_abs_error_dest_legit": round(float(np.abs(err_dest[legit_mask]).mean()), 2),
            },
        }

    def filter_and_subsample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1 of Recommended Pipeline:
        Restrict the modeling dataset to `type == 'TRANSFER'` and `type == 'CASH_OUT'`,
        and optionally perform stratified negative subsampling while keeping 100% of fraud cases.
        """
        valid_types = self.remediation.get("valid_fraud_types", ["TRANSFER", "CASH_OUT"])
        if valid_types:
            df_filtered = df.loc[df["type"].isin(valid_types)].copy()
        else:
            df_filtered = df.copy()

        sample_neg = self.remediation.get("sample_negative_count", None)
        seed = self.remediation.get("random_seed", 42)

        if sample_neg is not None:
            pos_df = df_filtered.loc[df_filtered["isFraud"] == 1]
            neg_df = df_filtered.loc[df_filtered["isFraud"] == 0]
            if len(neg_df) > sample_neg:
                neg_sampled = neg_df.sample(n=int(sample_neg), random_state=seed)
                df_filtered = (
                    pd.concat([pos_df, neg_sampled], axis=0)
                    .sample(frac=1.0, random_state=seed)
                    .reset_index(drop=True)
                )

        return df_filtered

    def prepare_modeling_dataframe(
        self, df: pd.DataFrame, fix_leakage: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
        """
        Step 2 & 3 of Recommended Pipeline:
        - If `fix_leakage=True`, recalculates and overwrites `newbalanceOrig` and `newbalanceDest`
          using `oldbalance` and `amount` to eliminate the PaySim Ledger Bug, and derives
          behavioral ratios and micro-seasonality features.
        - If `fix_leakage=False` (used only for the leakage ablation demonstration),
          retains raw `newbalanceOrig`, `newbalanceDest`, and raw discrepancy errors to show
          how severely a naive model cheats on the simulation bug.
        """
        work = df.copy()
        y = work["isFraud"].astype(int).reset_index(drop=True)

        amount = work["amount"].astype(float)
        old_orig = work["oldbalanceOrg"].astype(float)
        old_dest = work["oldbalanceDest"].astype(float)

        if fix_leakage:
            # Programmatically recalculate and overwrite post-transaction balances
            # to enforce a mathematically sound double-entry ledger:
            new_orig = np.maximum(0.0, old_orig - amount)
            new_dest = old_dest + amount
        else:
            new_orig = work["newbalanceOrig"].astype(float)
            new_dest = work["newbalanceDest"].astype(float)

        # Micro-seasonality features (31-day / 744-hour simulation horizon)
        step = work["step"].astype(int)
        hour_of_day = step % 24
        day_of_week = (step // 24) % 7

        # Behavioral Ratio Features:
        # 1. Account Depletion Ratio = amount / oldbalanceOrg
        #    When oldbalanceOrg == 0 and amount > 0 (anomalous overdraft/empty-origin transfer),
        #    we cap the ratio at a distinct high value (e.g., 2.0) and provide a dedicated binary flag
        #    so the ratio remains bounded and interpretable.
        account_depletion_ratio = np.where(
            old_orig > 0.0,
            np.minimum(amount / old_orig, 5.0),
            0.0,
        )
        is_zero_oldbalance_orig = (old_orig == 0.0).astype(float)
        is_zero_oldbalance_dest = (old_dest == 0.0).astype(float)

        # 2. Destination Relative Surge = amount / (oldbalanceDest + amount + 1.0)
        dest_surge_ratio = amount / (old_dest + amount + 1.0)

        # 3. Transaction Type indicator (1 = TRANSFER, 0 = CASH_OUT)
        is_transfer = (work["type"] == "TRANSFER").astype(float)

        features_dict = {
            "amount": amount.values,
            "oldbalanceOrg": old_orig.values,
            "newbalanceOrig": new_orig.values,
            "oldbalanceDest": old_dest.values,
            "newbalanceDest": new_dest.values,
            "account_depletion_ratio": account_depletion_ratio,
            "dest_surge_ratio": dest_surge_ratio,
            "is_zero_oldbalance_orig": is_zero_oldbalance_orig.values,
            "is_zero_oldbalance_dest": is_zero_oldbalance_dest.values,
            "hour_of_day": hour_of_day.values.astype(float),
            "day_of_week": day_of_week.values.astype(float),
            "is_transfer": is_transfer.values,
        }

        feature_descriptions = {
            "amount": "Transaction amount in local currency",
            "oldbalanceOrg": "Origin account initial balance before transaction",
            "newbalanceOrig": "Origin account balance after transaction (recalculated ledger)",
            "oldbalanceDest": "Destination account initial balance before transaction",
            "newbalanceDest": "Destination account balance after transaction (recalculated ledger)",
            "account_depletion_ratio": "Ratio of transaction amount to origin balance (amount / oldbalanceOrg, capped at 5.0)",
            "dest_surge_ratio": "Ratio of transaction amount to post-transaction destination balance",
            "is_zero_oldbalance_orig": "Indicator (1/0) whether origin account started with zero balance",
            "is_zero_oldbalance_dest": "Indicator (1/0) whether destination account started with zero balance",
            "hour_of_day": "Micro-seasonality: Hour of day (step mod 24)",
            "day_of_week": "Micro-seasonality: Day of week (floor(step / 24) mod 7)",
            "is_transfer": "Transaction type indicator (1 = TRANSFER, 0 = CASH_OUT)",
        }

        if not fix_leakage:
            # Include the raw ledger discrepancies to demonstrate target leakage in ablation study
            features_dict["raw_error_orig"] = ((old_orig - amount) - new_orig).values
            features_dict["raw_error_dest"] = ((old_dest + amount) - new_dest).values
            feature_descriptions["raw_error_orig"] = "LEAKED: Raw origin ledger discrepancy"
            feature_descriptions["raw_error_dest"] = "LEAKED: Raw destination ledger discrepancy"

        X = pd.DataFrame(features_dict)
        return X, y, feature_descriptions


class GenericDatasetAdapter(BaseDatasetAdapter):
    """
    Fallback adapter for any future tabular fraud dataset specified via YAML config.
    """

    def load_raw_data(self) -> pd.DataFrame:
        csv_path = Path(self.config["raw_csv_path"])
        return pd.read_csv(csv_path)

    def compute_domain_audit_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        target = self.target_col
        return {
            "total_rows": int(len(df)),
            "total_fraud": int(df[target].sum()),
            "global_fraud_rate_pct": round(100.0 * float(df[target].mean()), 4),
        }

    def filter_and_subsample(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def prepare_modeling_dataframe(
        self, df: pd.DataFrame, fix_leakage: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
        work = df.copy()
        y = work[self.target_col].astype(int).reset_index(drop=True)
        drop_cols = set(self.schema.get("drop_cols", [])) | {self.target_col}
        keep_cols = [c for c in work.columns if c not in drop_cols]
        X = pd.get_dummies(work[keep_cols], drop_first=True).astype(float).reset_index(drop=True)
        descriptions = {col: f"Feature {col}" for col in X.columns}
        return X, y, descriptions


def get_dataset_adapter(config: Dict[str, Any]) -> BaseDatasetAdapter:
    """Factory function returning the configured DatasetAdapter instance."""
    adapter_type = config.get("adapter_type", "paysim").lower()
    if adapter_type == "paysim":
        return PaySimAdapter(config)
    return GenericDatasetAdapter(config)
