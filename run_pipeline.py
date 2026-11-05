"""
Main Modular CLI Entrypoint (`run_pipeline.py`).

Allows running the entire fraud detection & custom SHAP pipeline end-to-end or
executing any individual stage independently when switching datasets or parameters:

Examples:
    python run_pipeline.py --config configs/paysim_config.yaml --stage all
    python run_pipeline.py --config configs/paysim_config.yaml --stage 1
    python run_pipeline.py --config configs/paysim_config.yaml --stage 2
    python run_pipeline.py --config configs/paysim_config.yaml --stage 3
    python run_pipeline.py --config configs/paysim_config.yaml --stage 4
    python run_pipeline.py --config configs/paysim_config.yaml --stage 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.stage1_audit_outliers import run_stage1
from src.stage2_remediate_features import run_stage2
from src.stage3_training import run_stage3
from src.stage4_shap_custom import run_stage4_custom
from src.stage4_shap_library import run_stage4_library
from src.stage5_reporting import run_stage5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modular Financial Fraud Detection & Custom TreeSHAP Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paysim_config.yaml",
        help="Path to the dataset YAML configuration file.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4", "4a", "4b", "5"],
        help="Which independent pipeline stage to execute ('1'..'5' or 'all').",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)

    if args.stage in ("all", "1"):
        run_stage1(cfg_path)
    if args.stage in ("all", "2"):
        run_stage2(cfg_path)
    if args.stage in ("all", "3"):
        run_stage3(cfg_path)
    if args.stage in ("all", "4", "4a"):
        run_stage4_custom(cfg_path)
    if args.stage in ("all", "4", "4b"):
        run_stage4_library(cfg_path)
    if args.stage in ("all", "5"):
        run_stage5(cfg_path)


if __name__ == "__main__":
    main()
