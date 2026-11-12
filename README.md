# Transaction Anomaly Detection & Custom TreeSHAP Explainability

A modular, dataset-agnostic machine learning pipeline for detecting fraudulent financial transactions and explaining model decisions using **from-scratch TreeSHAP** (for XGBoost) and the official `shap` library (for LightGBM).

Initially calibrated for the **PaySim Synthetic Mobile Money Dataset**, the pipeline explicitly audits and eliminates PaySim's synthetic target leakage (**"The Ledger Bug"**) before training cost-sensitive classifiers optimized for **PR-AUC (AUPRC)** and asymmetric financial risk.

---

## Key Features

- **Modular, Dataset-Agnostic Architecture:** Every pipeline stage communicates through YAML configurations (`configs/`), an abstract `BaseDatasetAdapter` contract (`src/dataset_adapter.py`), and standardized Parquet/JSON artifacts. Switching datasets requires only adding a new config or adapter subclass.
- **Data Validity & Leakage Audit (Stage 1):** Checks schema integrity (nulls, negatives, infinities, duplicates), verifies domain constraints, evaluates the naive `isFlaggedFraud` rule, quantifies PaySim's destination-balance freeze bug, and runs univariate (IQR / Modified Z-Score) and multivariate (`IsolationForest`) outlier analysis.
- **Ledger Remediation & Behavioral Features (Stage 2):** Filters to fraud-eligible transaction types (`TRANSFER` and `CASH_OUT`), programmatically recalculates and overwrites `newbalanceOrig` and `newbalanceDest` to eliminate target leakage, and engineers behavioral ratios (`account_depletion_ratio`, `dest_surge_ratio`) and micro-seasonality (`hour_of_day`, `day_of_week`).
- **Cost-Sensitive Dual Modeling & Ablation (Stage 3):** Trains **XGBoost** and **LightGBM** with class-imbalance weighting (`scale_pos_weight`), optimizes thresholds for PR-AUC and asymmetric financial cost (`FN = $500`, `FP = $10`), and runs a leaked-vs-remediated ablation benchmark.
- **Hand-Implemented TreeSHAP + Library SHAP (Stage 4):**
  - **XGBoost (`src/stage4_shap_custom.py`):** Exact coalitional TreeSHAP implemented from scratch in Python/NumPy by parsing XGBoost JSON trees, verified against the Shapley Efficiency Axiom ($\phi_0 + \sum \phi_i(x) = f(x)$) to floating-point precision ($1.95 \times 10^{-14}$).
  - **LightGBM (`src/stage4_shap_library.py`):** Official `shap.TreeExplainer` benchmark and cross-model rank correlation analysis.
- **Automated Reporting (Stage 5):** Compiles all metrics, case studies, and diagnostic figures into `reports/final_report.md` and a self-contained `reports/final_report.html`.

---

## Project Structure

```text
Transaction_anomaly_detection/
├── archive/                        # Raw CSV dataset (gitignored; place paysim dataset.csv here)
├── configs/
│   ├── paysim_config.yaml          # PaySim schema, remediation rules & hyperparameters
│   └── template_config.yaml        # Template for adding custom datasets
├── data/
│   ├── 01_audit/                   # Stage 1 audit_summary.json
│   └── 02_modeling/                # Stage 2 Parquet train/test splits & feature_metadata.json
├── models/                         # Saved models, evaluation metrics & SHAP JSON outputs
├── reports/
│   ├── figures/                    # Diagnostic PNG charts
│   ├── final_report.md             # Generated Markdown report
│   └── final_report.html           # Self-contained HTML report with embedded plots
├── src/
│   ├── dataset_adapter.py          # BaseDatasetAdapter, PaySimAdapter, GenericDatasetAdapter
│   ├── stage1_audit_outliers.py    # Validity, leakage audit & outlier detection
│   ├── stage2_remediate_features.py# Ledger fix, behavioral features & Parquet splitting
│   ├── stage3_training.py          # XGBoost, LightGBM, PR-AUC & cost-matrix evaluation
│   ├── stage4_shap_custom.py       # Hand-implemented TreeSHAP for XGBoost
│   ├── stage4_shap_library.py      # Official shap library explainer for LightGBM
│   └── stage5_reporting.py         # Markdown & HTML report compiler
├── tests/
│   └── test_pipeline.py            # Pytest suite verifying contracts, metrics & SHAP axioms
├── run_pipeline.py                 # CLI entrypoint for running all or individual stages
├── PROJECT_DOCUMENTATION.md        # Detailed architecture & programming documentation
└── requirements.txt
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pozitron7/Transaction_anomaly_detection.git
   cd Transaction_anomaly_detection
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Provide the dataset:**
   Place `paysim dataset.csv` inside `archive/` (matching `raw_csv_path` in `configs/paysim_config.yaml`).

---

## Usage

Run the full pipeline or any individual stage via `run_pipeline.py`:

```bash
# Run the complete end-to-end pipeline (Stages 1 through 5)
python run_pipeline.py --config configs/paysim_config.yaml --stage all

# Run individual stages independently
python run_pipeline.py --config configs/paysim_config.yaml --stage 1   # Validity, leakage audit & outliers
python run_pipeline.py --config configs/paysim_config.yaml --stage 2   # Ledger remediation & Parquet splits
python run_pipeline.py --config configs/paysim_config.yaml --stage 3   # Model training & evaluation
python run_pipeline.py --config configs/paysim_config.yaml --stage 4a  # Hand-implemented XGBoost TreeSHAP
python run_pipeline.py --config configs/paysim_config.yaml --stage 4b  # Library LightGBM SHAP
python run_pipeline.py --config configs/paysim_config.yaml --stage 4   # Both 4a and 4b
python run_pipeline.py --config configs/paysim_config.yaml --stage 5   # Generate Markdown & HTML reports
```

### Using a Different Dataset
1. Copy `configs/template_config.yaml` to `configs/<your_dataset>.yaml` and update `raw_csv_path` and `schema`.
2. If your dataset requires custom domain rules, subclass `BaseDatasetAdapter` in `src/dataset_adapter.py` and register it in `get_dataset_adapter()`.
3. Run `python run_pipeline.py --config configs/<your_dataset>.yaml --stage all`.

---

## Results Summary (PaySim Remediated Ledger)

| Model | Test PR-AUC | Test ROC-AUC | Optimal F1 | Min Asymmetric Cost (`FN=$500, FP=$10`) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | `0.99944` | `0.99997` | `0.9978` | `$1,690` |
| **LightGBM** | `0.99966` | `0.99997` | `0.9995` | `$1,000` |

- **Custom XGBoost TreeSHAP Efficiency Error:** `1.954e-14` max absolute error across 300 test samples.
- **Cross-Model SHAP Agreement (XGBoost Custom vs. LightGBM Library):** Spearman rank correlation `0.7063`.

---

## Testing

Run the automated test suite with `pytest`:

```bash
python -m pytest tests/test_pipeline.py -v
```

For full architectural diagrams, mathematical derivations, and module-level API details, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).
