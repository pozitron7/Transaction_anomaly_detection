r"""
Stage 4B: Official `shap` Library Implementation for the Secondary Model (`src/stage4_shap_library.py`).

This standalone module executes:
1. Library `shap.TreeExplainer` Attribution on Secondary Algorithm (LightGBM / Random Forest):
   - Loads `models/secondary_model.joblib` and the identical global sample and flagged fraud
     case-study indices selected in Stage 4A (`models/custom_xgboost_shap.json`).
   - Computes SHAP values using the official `shap` Python library.
2. Cross-Model & Cross-Implementation Explainability Comparison:
   - Compares our **Hand-Implemented XGBoost TreeSHAP** against **Library SHAP on LightGBM**
     across the exact same flagged fraudulent transactions and global feature importance rankings.
   - Computes Spearman rank correlation and Pearson correlation between global feature
     attributions of the two algorithms.
3. Visualizations & Persistence:
   - Generates global SHAP comparison charts and side-by-side local case-study explanations
     (`reports/figures/stage4_library_shap_and_comparison.png`).
   - Saves structured outputs to `models/library_secondary_shap.json`.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src.dataset_adapter import load_config


def extract_positive_class_shap(
    shap_values_raw: Any,
    expected_value_raw: Any,
) -> tuple[np.ndarray, float]:
    """
    Normalize outputs from `shap.TreeExplainer` across different `shap` / LightGBM / sklearn
    versions (which may return a 2D array in log-odds space or a list/3D array of [neg, pos]).
    """
    if isinstance(shap_values_raw, list):
        shap_arr = np.asarray(shap_values_raw[1], dtype=np.float64)
    else:
        arr = np.asarray(shap_values_raw, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[2] == 2:
            shap_arr = arr[:, :, 1]
        else:
            shap_arr = arr

    if isinstance(expected_value_raw, (list, np.ndarray)):
        ev_arr = np.asarray(expected_value_raw, dtype=np.float64).ravel()
        base_val = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
    else:
        base_val = float(expected_value_raw)

    return shap_arr, base_val


def generate_library_shap_plots(
    feature_names: List[str],
    xgb_global_mean: Dict[str, float],
    sec_global_mean: Dict[str, float],
    xgb_cases: List[Dict[str, Any]],
    sec_cases: List[Dict[str, Any]],
    sec_model_name: str,
    figures_dir: Path,
) -> Dict[str, str]:
    """
    Generate side-by-side comparison figures between:
    - Hand-Implemented TreeSHAP on XGBoost
    - Official `shap` Library on Secondary Model (LightGBM)
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------------------
    # Figure 1: Normalized Global Feature Attribution Comparison (XGB Custom vs LightGBM Library)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=140)

    xgb_total = sum(xgb_global_mean.values()) or 1.0
    sec_total = sum(sec_global_mean.values()) or 1.0

    # Order features by average relative importance across both models
    combined_score = {
        f: 0.5 * (xgb_global_mean[f] / xgb_total + sec_global_mean[f] / sec_total)
        for f in feature_names
    }
    ordered_feats = [k for k, _ in sorted(combined_score.items(), key=lambda kv: kv[1])]

    y_pos = np.arange(len(ordered_feats))
    width = 0.38
    xgb_norm = [100.0 * xgb_global_mean[f] / xgb_total for f in ordered_feats]
    sec_norm = [100.0 * sec_global_mean[f] / sec_total for f in ordered_feats]

    axes[0].barh(y_pos + width / 2, xgb_norm, width, color="#1f78b4", label="Hand-Implemented SHAP (XGBoost)")
    axes[0].barh(y_pos - width / 2, sec_norm, width, color="#33a02c", label=f"Library SHAP ({sec_model_name})")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(ordered_feats)
    axes[0].set_xlabel("Share of Total Mean |SHAP| Attribution (%)")
    axes[0].set_title("Global SHAP Comparison: Custom XGBoost vs. Library LightGBM", fontsize=11.5, fontweight="bold")
    axes[0].legend(loc="lower right", frameon=True)

    # Panel B: Side-by-side Local Attribution for Flagged Fraud Case #1
    c_xgb = xgb_cases[0]
    c_sec = sec_cases[0]
    top_feats = [
        k for k, _ in sorted(
            c_xgb["shap_contributions"].items(),
            key=lambda kv: abs(kv[1]) + abs(c_sec["shap_contributions"][kv[0]]),
            reverse=True,
        )[:8][::-1]
    ]
    y_c = np.arange(len(top_feats))
    v_xgb = [c_xgb["shap_contributions"][f] for f in top_feats]
    v_sec = [c_sec["shap_contributions"][f] for f in top_feats]
    labels_c = [f"{f} ({c_xgb['feature_values'][f]:,.2f})" for f in top_feats]

    axes[1].barh(y_c + width / 2, v_xgb, width, color="#1f78b4", label="Hand-Implemented SHAP (XGBoost)")
    axes[1].barh(y_c - width / 2, v_sec, width, color="#33a02c", label=f"Library SHAP ({sec_model_name})")
    axes[1].axvline(0.0, color="#222222", linewidth=1.1)
    axes[1].set_yticks(y_c)
    axes[1].set_yticklabels(labels_c, fontsize=9)
    axes[1].set_xlabel("Local SHAP Contribution φ_i (Log-Odds)")
    axes[1].set_title(
        f"Flagged Fraud Case #1 (Row #{c_xgb['test_row_index']}) Cross-Model Agreement",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[1].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    comp_path = figures_dir / "stage4_library_shap_and_comparison.png"
    fig.savefig(comp_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "library_shap_and_comparison_plot": str(comp_path),
    }


def run_stage4_library(config_path: str | Path = "configs/paysim_config.yaml") -> Dict[str, Any]:
    """
    Execute Stage 4B: Compute official `shap` library attributions for the secondary
    classifier and compare them against our hand-implemented XGBoost SHAP outputs.
    """
    t0 = time.time()
    config = load_config(config_path)
    modeling_dir = Path(config["paths"]["modeling_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])

    print("[Stage 4B - Library SHAP] Loading test split, secondary model, and Stage 4A custom SHAP results...")
    X_test = pd.read_parquet(modeling_dir / "X_test.parquet")
    y_test = pd.read_parquet(modeling_dir / "y_test.parquet")["isFraud"].to_numpy(dtype=int)
    sec_model = joblib.load(models_dir / "secondary_model.joblib")

    with open(models_dir / "custom_xgboost_shap.json", "r", encoding="utf-8") as f:
        custom_xgb_data = json.load(f)

    feature_names = list(X_test.columns)
    global_sample_idx = custom_xgb_data["global_sample_indices"]
    case_indices = custom_xgb_data["selected_case_study_indices"]

    X_global_sample = X_test.iloc[global_sample_idx].reset_index(drop=True)
    X_cases = X_test.iloc[case_indices].reset_index(drop=True)

    sec_model_name = type(sec_model).__name__
    print(f"[Stage 4B - Library SHAP] Running official shap.TreeExplainer on {sec_model_name}...")
    lib_explainer = shap.TreeExplainer(sec_model)

    raw_global_shap = lib_explainer.shap_values(X_global_sample)
    global_shap_matrix, base_val = extract_positive_class_shap(
        raw_global_shap, lib_explainer.expected_value
    )

    raw_cases_shap = lib_explainer.shap_values(X_cases)
    cases_shap_matrix, _ = extract_positive_class_shap(
        raw_cases_shap, lib_explainer.expected_value
    )

    # Global Mean |SHAP| for Secondary Model
    sec_mean_abs = {
        col: round(float(np.abs(global_shap_matrix[:, i]).mean()), 5)
        for i, col in enumerate(feature_names)
    }
    sec_mean_abs_sorted = dict(
        sorted(sec_mean_abs.items(), key=lambda kv: kv[1], reverse=True)
    )

    # Format Case Study breakdowns for Secondary Model
    sec_probs = sec_model.predict_proba(X_cases)[:, 1]
    sec_cases: List[Dict[str, Any]] = []
    for idx, row_idx in enumerate(case_indices):
        phi_vec = cases_shap_matrix[idx]
        pred_margin = float(base_val + np.sum(phi_vec))
        feat_vals = {col: round(float(X_cases.iloc[idx][col]), 4) for col in feature_names}
        shap_contribs = {col: round(float(phi_vec[i]), 5) for i, col in enumerate(feature_names)}
        top_drivers = sorted(shap_contribs.items(), key=lambda kv: kv[1], reverse=True)[:3]

        sec_cases.append({
            "test_row_index": int(row_idx),
            "true_label": int(y_test[row_idx]),
            "predicted_probability": round(float(sec_probs[idx]), 5),
            "predicted_log_odds": round(pred_margin, 5),
            "base_value_phi0": round(base_val, 5),
            "feature_values": feat_vals,
            "shap_contributions": shap_contribs,
            "top_positive_fraud_drivers": [
                {"feature": k, "shap_log_odds": v, "feature_value": feat_vals[k]}
                for k, v in top_drivers
            ],
        })

    # Compute Cross-Model Rank & Linear Correlation on Global Mean |SHAP|
    xgb_mean_dict = custom_xgb_data["global_mean_abs_shap"]
    xgb_vec = np.array([xgb_mean_dict[f] for f in feature_names])
    sec_vec = np.array([sec_mean_abs[f] for f in feature_names])
    spearman_corr, _ = spearmanr(xgb_vec, sec_vec)
    pearson_corr = float(np.corrcoef(xgb_vec, sec_vec)[0, 1])

    plot_paths = generate_library_shap_plots(
        feature_names=feature_names,
        xgb_global_mean=xgb_mean_dict,
        sec_global_mean=sec_mean_abs,
        xgb_cases=custom_xgb_data["case_studies"],
        sec_cases=sec_cases,
        sec_model_name=sec_model_name,
        figures_dir=figures_dir,
    )

    results = {
        "algorithm": f"Official `shap.TreeExplainer` Library on {sec_model_name}",
        "base_value_phi0_log_odds": round(base_val, 5),
        "global_sample_size": int(len(X_global_sample)),
        "global_mean_abs_shap": sec_mean_abs_sorted,
        "cross_model_shap_agreement": {
            "spearman_rank_correlation": round(float(spearman_corr), 4),
            "pearson_linear_correlation": round(float(pearson_corr), 4),
        },
        "case_studies": sec_cases,
        "plots": plot_paths,
        "execution_time_sec": round(time.time() - t0, 2),
    }

    out_path = models_dir / "library_secondary_shap.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"[Stage 4B - Library SHAP] Complete in {results['execution_time_sec']}s "
        f"(Cross-model Spearman r={spearman_corr:.4f}) -> saved {out_path}"
    )
    return results


if __name__ == "__main__":
    run_stage4_library()
