"""
Unit and integration tests for the Modular Fraud Detection & Custom TreeSHAP Pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_adapter import get_dataset_adapter, load_config
from src.stage4_shap_custom import CustomXGBoostTreeSHAPExplainer
import xgboost as xgb


@pytest.fixture(scope="session")
def config():
    return load_config("configs/paysim_config.yaml")


def test_config_loader(config):
    assert "schema" in config
    assert "modeling" in config
    assert "paths" in config
    assert config["dataset_name"] == "paysim"


def test_dataset_adapter_interface(config):
    adapter = get_dataset_adapter(config)
    assert adapter.target_col == "isFraud"
    assert hasattr(adapter, "load_raw_data")
    assert hasattr(adapter, "compute_domain_audit_metrics")
    assert hasattr(adapter, "prepare_modeling_dataframe")


def test_stage1_audit_output(config):
    audit_file = Path(config["paths"]["audit_dir"]) / "audit_summary.json"
    assert audit_file.exists(), f"{audit_file} missing"
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["validity_checks"]["is_valid_schema"] is True
    assert data["validity_checks"]["total_null_values"] == 0
    assert data["validity_checks"]["exact_duplicate_rows"] == 0
    assert "TRANSFER" in data["domain_audit"]["type_breakdown"]
    assert "CASH_OUT" in data["domain_audit"]["type_breakdown"]
    # Check that Ledger Bug was quantified
    assert "ledger_bug_and_leakage_audit" in data["domain_audit"]
    assert data["domain_audit"]["ledger_bug_and_leakage_audit"]["dest_balance_unchanged_in_fraud_pct"] > 40.0


def test_stage2_data_splits(config):
    modeling_dir = Path(config["paths"]["modeling_dir"])
    meta_file = modeling_dir / "feature_metadata.json"
    assert meta_file.exists()

    X_train = pd.read_parquet(modeling_dir / "X_train.parquet")
    X_test = pd.read_parquet(modeling_dir / "X_test.parquet")
    y_train = pd.read_parquet(modeling_dir / "y_train.parquet")
    y_test = pd.read_parquet(modeling_dir / "y_test.parquet")

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train.columns) == 12
    assert "account_depletion_ratio" in X_train.columns
    # Ensure no NaN or Inf in feature matrices
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_test.to_numpy()).all()


def test_stage3_model_evaluations(config):
    models_dir = Path(config["paths"]["models_dir"])
    eval_file = models_dir / "evaluation_metrics.json"
    assert eval_file.exists()

    with open(eval_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Verify both models evaluated on PR-AUC
    xgb_pr = metrics["xgboost_remediated"]["metrics"]["pr_auc"]
    sec_pr = metrics["secondary_model_remediated"]["metrics"]["pr_auc"]
    assert xgb_pr > 0.95
    assert sec_pr > 0.95


def test_stage4_custom_treeshap_efficiency(config):
    models_dir = Path(config["paths"]["models_dir"])
    shap_file = models_dir / "custom_xgboost_shap.json"
    assert shap_file.exists()

    with open(shap_file, "r", encoding="utf-8") as f:
        c_shap = json.load(f)

    # Verify efficiency axiom holds within numerical tolerance
    verif = c_shap["mathematical_verification"]
    assert verif["verified_exact"] is True
    assert verif["max_abs_error"] < 1e-10

    # Test custom explainer live on a small synthetic instance
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(models_dir / "xgboost_model.json")
    feature_names = list(c_shap["global_mean_abs_shap"].keys())

    explainer = CustomXGBoostTreeSHAPExplainer(xgb_model, feature_names)
    assert len(explainer.trees) == 250

    sample_x = np.ones(len(feature_names), dtype=np.float64)
    phi, phi0, margin = explainer.explain_instance(sample_x)
    assert np.isclose(phi0 + np.sum(phi), margin, atol=1e-10)


def test_stage5_reports_exist(config):
    reports_dir = Path(config["paths"]["reports_dir"])
    md_report = reports_dir / "final_report.md"
    html_report = reports_dir / "final_report.html"

    assert md_report.exists()
    assert html_report.exists()
    assert md_report.stat().st_size > 5000
    assert html_report.stat().st_size > 50000  # Embedded Base64 images


def test_spark_data_handling_unit(config):
    from src.spark_utils import get_spark_session
    from src.stage1_audit_outliers import compute_validity_checks

    spark = get_spark_session(config)
    try:
        adapter = get_dataset_adapter(config)
        rows = [
            (1, "TRANSFER", 100.0, "C1", 100.0, 0.0, "C2", 0.0, 0.0, 1, 0),
            (2, "CASH_OUT", 50.0, "C3", 200.0, 150.0, "C4", 10.0, 60.0, 0, 0),
            (3, "PAYMENT", 25.0, "C5", 100.0, 75.0, "M1", 0.0, 0.0, 0, 0),
        ]
        df = spark.createDataFrame(rows, schema=adapter.PAYSIM_SCHEMA)

        validity = compute_validity_checks(df, config["schema"]["numeric_cols"])
        assert validity["is_valid_schema"] is True
        assert validity["total_null_values"] == 0
        assert validity["exact_duplicate_rows"] == 0

        domain = adapter.compute_domain_audit_metrics(df)
        assert domain["total_rows"] == 3
        assert domain["total_fraud"] == 1
        assert domain["ledger_bug_and_leakage_audit"]["dest_balance_unchanged_in_fraud_pct"] == 100.0

        filtered = adapter.filter_and_subsample(df)
        assert filtered.count() == 2  # PAYMENT excluded

        clean_df, feat_cols, _ = adapter.prepare_modeling_dataframe(filtered, fix_leakage=True)
        assert len(feat_cols) == 12
        fraud_row = clean_df.filter("isFraud == 1").collect()[0]
        assert fraud_row["newbalanceDest"] == 100.0  # Ledger bug fixed (0 + 100)

        leaked_df, leaked_cols, _ = adapter.prepare_modeling_dataframe(filtered, fix_leakage=False)
        assert len(leaked_cols) == 14
        assert "raw_error_dest" in leaked_cols
    finally:
        spark.stop()
