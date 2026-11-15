"""
Dataset Adapter & Configuration Layer (Apache Spark Backend).

This module provides a clean abstraction between dataset-specific schemas/quirks
(such as the PaySim synthetic dataset and its known Ledger Bug) and downstream
modeling, explainability, and reporting stages.

All large-scale ingestion, domain auditing, filtering, subsampling, ledger
remediation, and feature engineering are executed natively in Apache Spark
(`pyspark.sql`) so the pipeline scales out-of-core to extremely large datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType


def load_config(config_path: str | Path = "configs/paysim_config.yaml") -> Dict[str, Any]:
    """Load and validate the YAML pipeline configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure output directories exist
    for _, dir_val in config.get("paths", {}).items():
        Path(dir_val).mkdir(parents=True, exist_ok=True)

    return config


def _read_spark_source(
    spark: SparkSession, path: Path, schema: StructType | None = None
) -> DataFrame:
    """Read a CSV or Parquet dataset into a Spark DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path.resolve()}")

    path_str = str(path)
    if path.suffix.lower() == ".parquet" or (
        path.is_dir() and any(path.glob("*.parquet"))
    ):
        return spark.read.parquet(path_str)

    reader = spark.read.option("header", "true")
    if schema is not None:
        return reader.schema(schema).csv(path_str)
    return reader.option("inferSchema", "true").csv(path_str)


class BaseDatasetAdapter(ABC):
    """
    Abstract base class defining the contract for any fraud detection dataset.
    Operates on Spark DataFrames for scalable distributed processing.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.schema = config["schema"]
        self.remediation = config.get("remediation", {})

    @property
    def target_col(self) -> str:
        return self.schema["target_col"]

    @abstractmethod
    def load_raw_data(self, spark: SparkSession) -> DataFrame:
        """Load the raw dataset into a Spark DataFrame using an explicit schema."""
        pass

    @abstractmethod
    def compute_domain_audit_metrics(self, df: DataFrame) -> Dict[str, Any]:
        """
        Execute dataset-specific validity checks and artifact/leakage audits in Spark.
        Returns a JSON-serializable dictionary of findings.
        """
        pass

    @abstractmethod
    def filter_and_subsample(self, df: DataFrame) -> DataFrame:
        """Filter domain-valid rows and perform stratified negative subsampling in Spark."""
        pass

    @abstractmethod
    def prepare_modeling_dataframe(
        self, df: DataFrame, fix_leakage: bool = True
    ) -> Tuple[DataFrame, List[str], Dict[str, str]]:
        """
        Filter, remediate target leakage, and engineer behavioral features in Spark.

        Returns:
            modeling_df: Spark DataFrame containing `feature_cols + [target_col]`
                         (plus `_is_test` if already attached).
            feature_cols: Ordered list of numeric feature column names.
            feature_descriptions: Mapping from column name to human-readable description.
        """
        pass


class PaySimAdapter(BaseDatasetAdapter):
    """
    Spark-native Adapter for the PaySim Synthetic Mobile Money Dataset.

    Explicitly accounts for PaySim's architectural properties and limitations:
    1. Categorical Constraint: Fraud only exists in TRANSFER and CASH_OUT.
    2. Target Leakage (The Ledger Bug): Fraudulent transactions often leave
       destination balances at 0.0 or unchanged. We audit this correlation in Stage 1
       and programmatically recalculate `newbalanceOrig` and `newbalanceDest` in Stage 2.
    3. Behavioral Ratios & Micro-Seasonality: Derives account depletion ratios
       and hour-of-day/day-of-week features over the 744-hour (31-day) simulation.
    """

    PAYSIM_SCHEMA = StructType(
        [
            StructField("step", IntegerType(), True),
            StructField("type", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("nameOrig", StringType(), True),
            StructField("oldbalanceOrg", DoubleType(), True),
            StructField("newbalanceOrig", DoubleType(), True),
            StructField("nameDest", StringType(), True),
            StructField("oldbalanceDest", DoubleType(), True),
            StructField("newbalanceDest", DoubleType(), True),
            StructField("isFraud", IntegerType(), True),
            StructField("isFlaggedFraud", IntegerType(), True),
        ]
    )

    def load_raw_data(self, spark: SparkSession) -> DataFrame:
        csv_path = Path(self.config["raw_csv_path"])
        return _read_spark_source(spark, csv_path, schema=self.PAYSIM_SCHEMA)

    def compute_domain_audit_metrics(self, df: DataFrame) -> Dict[str, Any]:
        """
        Quantify PaySim's structural properties, naive rule performance, entity prefixes,
        and the exact severity of the Ledger Bug target leakage using distributed Spark
        aggregations.
        """
        is_tc = F.col("type").isin(["TRANSFER", "CASH_OUT"])

        # Single-pass global, naive-flag, and entity-prefix aggregations
        global_row = df.agg(
            F.count("*").alias("total_rows"),
            F.sum(F.col("isFraud").cast("long")).alias("total_fraud"),
            F.min("step").alias("step_min"),
            F.max("step").alias("step_max"),
            F.sum(F.col("isFlaggedFraud").cast("long")).alias("flagged_total"),
            F.sum(
                F.when((F.col("isFlaggedFraud") == 1) & (F.col("isFraud") == 1), 1).otherwise(0)
            ).alias("flagged_true_pos"),
            F.sum(
                F.when((F.col("isFlaggedFraud") == 1) & (F.col("isFraud") == 0), 1).otherwise(0)
            ).alias("flagged_false_pos"),
            F.sum(F.when(F.col("nameOrig").startswith("M"), 1).otherwise(0)).alias("orig_merchants"),
            F.sum(F.when(F.col("nameDest").startswith("M"), 1).otherwise(0)).alias("dest_merchants"),
            F.sum(F.when(is_tc & F.col("nameDest").startswith("M"), 1).otherwise(0)).alias(
                "dest_merchants_tc"
            ),
        ).collect()[0]

        total_rows = int(global_row["total_rows"] or 0)
        total_fraud = int(global_row["total_fraud"] or 0)
        flagged_total = int(global_row["flagged_total"] or 0)
        flagged_true_pos = int(global_row["flagged_true_pos"] or 0)
        flagged_false_pos = int(global_row["flagged_false_pos"] or 0)
        flagged_false_neg = total_fraud - flagged_true_pos

        # Categorical constraint breakdown across all 5 transaction types
        type_rows = (
            df.groupBy("type")
            .agg(
                F.count("*").alias("total_tx"),
                F.sum(F.col("isFraud").cast("long")).alias("fraud_tx"),
                F.mean("amount").alias("mean_amount"),
            )
            .collect()
        )
        type_breakdown: Dict[str, Any] = {}
        for row in type_rows:
            t_name = str(row["type"])
            cnt = int(row["total_tx"] or 0)
            f_cnt = int(row["fraud_tx"] or 0)
            type_breakdown[t_name] = {
                "total_transactions": cnt,
                "share_of_dataset_pct": round(100.0 * cnt / max(total_rows, 1), 3),
                "fraud_transactions": f_cnt,
                "fraud_rate_pct": round(100.0 * f_cnt / max(cnt, 1), 4),
                "mean_amount": round(float(row["mean_amount"] or 0.0), 2),
            }

        # Quantify "The Ledger Bug" on TRANSFER and CASH_OUT transactions in a single pass
        tc_df = (
            df.filter(is_tc)
            .withColumn(
                "abs_err_orig",
                F.abs((F.col("oldbalanceOrg") - F.col("amount")) - F.col("newbalanceOrig")),
            )
            .withColumn(
                "abs_err_dest",
                F.abs((F.col("oldbalanceDest") + F.col("amount")) - F.col("newbalanceDest")),
            )
            .withColumn(
                "dest_unchanged",
                ((F.col("newbalanceDest") == F.col("oldbalanceDest")) & (F.col("amount") > 0)).cast(
                    "double"
                ),
            )
            .withColumn(
                "dest_zero_after",
                (
                    (F.col("oldbalanceDest") == 0.0)
                    & (F.col("newbalanceDest") == 0.0)
                    & (F.col("amount") > 0.0)
                ).cast("double"),
            )
            .withColumn(
                "exact_depletion",
                (
                    (
                        F.abs(F.col("amount") - F.col("oldbalanceOrg"))
                        <= (1e-8 + 1e-5 * F.abs(F.col("oldbalanceOrg")))
                    )
                    & (F.col("amount") > 0.0)
                ).cast("double"),
            )
            .withColumn(
                "dest_eq_flag",
                (F.col("newbalanceDest") == F.col("oldbalanceDest")).cast("double"),
            )
            .withColumn("y_double", F.col("isFraud").cast("double"))
        )

        is_f = F.col("isFraud") == 1
        is_l = F.col("isFraud") == 0
        lb_row = tc_df.agg(
            F.count("*").alias("tc_rows"),
            F.mean(F.when(is_f, F.col("dest_unchanged"))).alias("dest_unchanged_fraud"),
            F.mean(F.when(is_l, F.col("dest_unchanged"))).alias("dest_unchanged_legit"),
            F.mean(F.when(is_f, F.col("dest_zero_after"))).alias("dest_zero_fraud"),
            F.mean(F.when(is_l, F.col("dest_zero_after"))).alias("dest_zero_legit"),
            F.mean(F.when(is_f, F.col("exact_depletion"))).alias("exact_depl_fraud"),
            F.mean(F.when(is_l, F.col("exact_depletion"))).alias("exact_depl_legit"),
            F.corr("abs_err_orig", "y_double").alias("corr_err_orig"),
            F.corr("abs_err_dest", "y_double").alias("corr_err_dest"),
            F.corr("dest_eq_flag", "y_double").alias("corr_dest_unchanged"),
            F.mean(F.when(is_f, F.col("abs_err_dest"))).alias("mean_err_dest_fraud"),
            F.mean(F.when(is_l, F.col("abs_err_dest"))).alias("mean_err_dest_legit"),
        ).collect()[0]

        return {
            "total_rows": total_rows,
            "total_fraud": total_fraud,
            "global_fraud_rate_pct": round(100.0 * total_fraud / max(total_rows, 1), 4),
            "step_min": int(global_row["step_min"] or 0),
            "step_max": int(global_row["step_max"] or 0),
            "type_breakdown": type_breakdown,
            "naive_isFlaggedFraud_audit": {
                "total_flagged": flagged_total,
                "true_positives": flagged_true_pos,
                "false_positives": flagged_false_pos,
                "false_negatives": flagged_false_neg,
                "recall_pct": round(100.0 * flagged_true_pos / max(total_fraud, 1), 4),
            },
            "entity_prefix_audit": {
                "orig_merchant_count": int(global_row["orig_merchants"] or 0),
                "dest_merchant_count": int(global_row["dest_merchants"] or 0),
                "dest_merchants_in_transfer_or_cashout": int(global_row["dest_merchants_tc"] or 0),
            },
            "ledger_bug_and_leakage_audit": {
                "transfer_cashout_rows": int(lb_row["tc_rows"] or 0),
                "dest_balance_unchanged_in_fraud_pct": round(
                    float(lb_row["dest_unchanged_fraud"] or 0.0) * 100.0, 2
                ),
                "dest_balance_unchanged_in_legit_pct": round(
                    float(lb_row["dest_unchanged_legit"] or 0.0) * 100.0, 2
                ),
                "dest_zero_before_and_after_in_fraud_pct": round(
                    float(lb_row["dest_zero_fraud"] or 0.0) * 100.0, 2
                ),
                "dest_zero_before_and_after_in_legit_pct": round(
                    float(lb_row["dest_zero_legit"] or 0.0) * 100.0, 2
                ),
                "exact_100pct_account_depletion_in_fraud_pct": round(
                    float(lb_row["exact_depl_fraud"] or 0.0) * 100.0, 2
                ),
                "exact_100pct_account_depletion_in_legit_pct": round(
                    float(lb_row["exact_depl_legit"] or 0.0) * 100.0, 2
                ),
                "corr_abs_error_orig_with_fraud": round(float(lb_row["corr_err_orig"] or 0.0), 4),
                "corr_abs_error_dest_with_fraud": round(float(lb_row["corr_err_dest"] or 0.0), 4),
                "corr_dest_unchanged_with_fraud": round(
                    float(lb_row["corr_dest_unchanged"] or 0.0), 4
                ),
                "mean_abs_error_dest_fraud": round(float(lb_row["mean_err_dest_fraud"] or 0.0), 2),
                "mean_abs_error_dest_legit": round(float(lb_row["mean_err_dest_legit"] or 0.0), 2),
            },
        }

    def filter_and_subsample(self, df: DataFrame) -> DataFrame:
        """
        Restrict the modeling dataset to `type == 'TRANSFER'` and `type == 'CASH_OUT'`,
        and optionally perform stratified negative subsampling in Spark while keeping
        100% of positive fraud cases.
        """
        valid_types = self.remediation.get("valid_fraud_types", ["TRANSFER", "CASH_OUT"])
        df_filtered = df.filter(F.col("type").isin(valid_types)) if valid_types else df

        sample_neg = self.remediation.get("sample_negative_count", None)
        seed = int(self.remediation.get("random_seed", 42))

        if sample_neg is not None:
            sample_neg_int = int(sample_neg)
            pos_df = df_filtered.filter(F.col("isFraud") == 1)
            neg_df = df_filtered.filter(F.col("isFraud") == 0)
            neg_count = neg_df.count()
            if neg_count > sample_neg_int:
                # Slight oversample factor + limit ensures exact count without collecting
                frac = min(1.0, (sample_neg_int * 1.05) / float(neg_count))
                neg_sampled = neg_df.sample(withReplacement=False, fraction=frac, seed=seed).limit(
                    sample_neg_int
                )
                df_filtered = pos_df.unionByName(neg_sampled)

        return df_filtered

    def prepare_modeling_dataframe(
        self, df: DataFrame, fix_leakage: bool = True
    ) -> Tuple[DataFrame, List[str], Dict[str, str]]:
        """
        Recalculate ledger balances (when `fix_leakage=True`) and engineer behavioral
        ratios & micro-seasonality features natively in Spark.
        """
        amount = F.col("amount").cast("double")
        old_orig = F.col("oldbalanceOrg").cast("double")
        old_dest = F.col("oldbalanceDest").cast("double")
        step = F.col("step").cast("long")

        if fix_leakage:
            new_orig_expr = F.greatest(F.lit(0.0), old_orig - amount)
            new_dest_expr = old_dest + amount
        else:
            new_orig_expr = F.col("newbalanceOrig").cast("double")
            new_dest_expr = F.col("newbalanceDest").cast("double")

        feature_cols = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "account_depletion_ratio",
            "dest_surge_ratio",
            "is_zero_oldbalance_orig",
            "is_zero_oldbalance_dest",
            "hour_of_day",
            "day_of_week",
            "is_transfer",
        ]

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

        work = (
            df.withColumn("amount", amount)
            .withColumn("oldbalanceOrg", old_orig)
            .withColumn("newbalanceOrig", new_orig_expr)
            .withColumn("oldbalanceDest", old_dest)
            .withColumn("newbalanceDest", new_dest_expr)
            .withColumn(
                "account_depletion_ratio",
                F.when(old_orig > 0.0, F.least(amount / old_orig, F.lit(5.0))).otherwise(0.0),
            )
            .withColumn("dest_surge_ratio", amount / (old_dest + amount + F.lit(1.0)))
            .withColumn(
                "is_zero_oldbalance_orig", F.when(old_orig == 0.0, 1.0).otherwise(0.0)
            )
            .withColumn(
                "is_zero_oldbalance_dest", F.when(old_dest == 0.0, 1.0).otherwise(0.0)
            )
            .withColumn("hour_of_day", (step % F.lit(24)).cast("double"))
            .withColumn("day_of_week", (F.floor(step / F.lit(24.0)) % F.lit(7)).cast("double"))
            .withColumn("is_transfer", F.when(F.col("type") == "TRANSFER", 1.0).otherwise(0.0))
            .withColumn("isFraud", F.col("isFraud").cast("int"))
        )

        if not fix_leakage:
            work = work.withColumn(
                "raw_error_orig", (old_orig - amount) - F.col("newbalanceOrig")
            ).withColumn(
                "raw_error_dest", (old_dest + amount) - F.col("newbalanceDest")
            )
            feature_cols.extend(["raw_error_orig", "raw_error_dest"])
            feature_descriptions["raw_error_orig"] = "LEAKED: Raw origin ledger discrepancy"
            feature_descriptions["raw_error_dest"] = "LEAKED: Raw destination ledger discrepancy"

        select_cols = list(feature_cols) + [self.target_col]
        if "_is_test" in work.columns:
            select_cols.append("_is_test")

        return work.select(*select_cols), feature_cols, feature_descriptions


class GenericDatasetAdapter(BaseDatasetAdapter):
    """
    Fallback Spark adapter for any future tabular fraud dataset specified via YAML config.
    """

    def load_raw_data(self, spark: SparkSession) -> DataFrame:
        csv_path = Path(self.config["raw_csv_path"])
        return _read_spark_source(spark, csv_path, schema=None)

    def compute_domain_audit_metrics(self, df: DataFrame) -> Dict[str, Any]:
        target = self.target_col
        row = df.agg(
            F.count("*").alias("total_rows"),
            F.sum(F.col(target).cast("long")).alias("total_fraud"),
            F.mean(F.col(target).cast("double")).alias("fraud_rate"),
        ).collect()[0]
        return {
            "total_rows": int(row["total_rows"] or 0),
            "total_fraud": int(row["total_fraud"] or 0),
            "global_fraud_rate_pct": round(100.0 * float(row["fraud_rate"] or 0.0), 4),
        }

    def filter_and_subsample(self, df: DataFrame) -> DataFrame:
        sample_neg = self.remediation.get("sample_negative_count", None)
        seed = int(self.remediation.get("random_seed", 42))
        target = self.target_col
        if sample_neg is not None:
            sample_neg_int = int(sample_neg)
            pos_df = df.filter(F.col(target) == 1)
            neg_df = df.filter(F.col(target) == 0)
            neg_count = neg_df.count()
            if neg_count > sample_neg_int:
                frac = min(1.0, (sample_neg_int * 1.05) / float(neg_count))
                neg_sampled = neg_df.sample(withReplacement=False, fraction=frac, seed=seed).limit(
                    sample_neg_int
                )
                return pos_df.unionByName(neg_sampled)
        return df

    def prepare_modeling_dataframe(
        self, df: DataFrame, fix_leakage: bool = True
    ) -> Tuple[DataFrame, List[str], Dict[str, str]]:
        target = self.target_col
        drop_cols = set(self.schema.get("drop_cols", [])) | {target, "_is_test"}
        numeric_cols = [c for c in self.schema.get("numeric_cols", []) if c not in drop_cols and c in df.columns]

        work = df.withColumn(target, F.col(target).cast("int"))
        for col_name in numeric_cols:
            work = work.withColumn(col_name, F.col(col_name).cast("double"))

        descriptions = {col: f"Feature {col}" for col in numeric_cols}
        select_cols = list(numeric_cols) + [target]
        if "_is_test" in work.columns:
            select_cols.append("_is_test")
        return work.select(*select_cols), numeric_cols, descriptions


def get_dataset_adapter(config: Dict[str, Any]) -> BaseDatasetAdapter:
    """Factory function returning the configured DatasetAdapter instance."""
    adapter_type = config.get("adapter_type", "paysim").lower()
    if adapter_type == "paysim":
        return PaySimAdapter(config)
    return GenericDatasetAdapter(config)
