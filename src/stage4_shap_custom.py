r"""
Stage 4A: Hand-Implemented SHAP Algorithm for XGBoost (`src/stage4_shap_custom.py`).

This module implements the exact **TreeSHAP (Path-Dependent Shapley Additive exPlanations)**
algorithm completely from scratch in Python/NumPy without calling `shap` or XGBoost's
built-in `pred_contribs`.

========================================================================================
MATHEMATICAL & ALGORITHMIC EXPLANATION OF OUR HAND-IMPLEMENTED TREESHAP
========================================================================================
For any prediction model $f(x)$ with $M$ features $N = \{1, \dots, M\}$, the Shapley value
$\phi_i(f, x)$ attributing the prediction $f(x) - E[f(X)]$ to feature $i$ is defined by
the classical game-theoretic formula:

$$\phi_i(f, x) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(M - |S| - 1)!}{M!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

where $f_x(S) = E[f(X) \mid X_S = x_S]$ is the expected model prediction conditioned on
the subset of known features $S$.

In a decision tree $t$, evaluating $f_x^{(t)}(S)$ works recursively from the root node:
- If the split feature $d_v$ at internal node $v$ is in the active coalition ($d_v \in S$),
  we deterministically follow the branch taken by $x_{d_v}$ (the "hot" child).
- If the split feature is NOT in the coalition ($d_v \notin S$), we take the cover-weighted
  average of both children:
  $$E[f_t(X) \mid \text{node } v] = \frac{R_{\text{left}}}{R_v} E[f_t(X) \mid \text{left}] + \frac{R_{\text{right}}}{R_v} E[f_t(X) \mid \text{right}]$$
  where $R_v$ is the training sample cover (`cover` in XGBoost's JSON dump) reaching node $v$.

Key Efficiency Optimization (Active Tree Feature Reduction):
A decision tree of depth $D \le 6$ contains at most $2^D - 1 \le 63$ internal nodes, and in
practice splits on only $k \le 8$ distinct features $U_t \subseteq N$. Any feature
$j \notin U_t$ never appears in tree $t$, so its marginal contribution $f_x^{(t)}(S \cup \{j\}) - f_x^{(t)}(S) = 0$
for all $S$, meaning $\phi_j^{(t)}(x) = 0$ identically! Furthermore, by the Shapley Dummy/Symmetry
axioms, the Shapley values $\phi_i^{(t)}(x)$ for the active features $i \in U_t$ depend ONLY on
coalitions $S \subseteq U_t$ of the $k = |U_t|$ active features in that tree:

$$\phi_i^{(t)}(x) = \sum_{S \subseteq U_t \setminus \{i\}} \frac{|S|!(k - |S| - 1)!}{k!} \left[ f_x^{(t)}(S \cup \{i\}) - f_x^{(t)}(S) \right]$$

We implement this exact active-feature coalitional TreeSHAP solver using vectorized NumPy bitmask
evaluation across all $2^k$ coalitions of each tree's active feature set $U_t$, guaranteeing:
1. **Exact mathematical satisfaction of the Efficiency (Local Accuracy) Axiom** down to
   64-bit floating-point machine epsilon ($\approx 10^{-14}$):
   $$\phi_0 + \sum_{i=1}^{M} \phi_i(x) = f(x) \quad \text{(in log-odds margin space)}$$
2. **Vectorized batch execution** across hundreds of test transactions in just a few seconds.
========================================================================================
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.dataset_adapter import load_config


@dataclass
class TreeNode:
    """Parsed node from an XGBoost JSON decision tree."""
    node_id: int
    is_leaf: bool
    leaf_value: float = 0.0
    split_feature: Optional[str] = None
    split_feature_idx: int = -1
    split_threshold: float = 0.0
    yes_id: int = -1
    no_id: int = -1
    missing_id: int = -1
    cover: float = 0.0
    yes_weight: float = 0.5
    no_weight: float = 0.5


class ParsedXGBTree:
    r"""
    Represents a single decision tree from an XGBoost ensemble with:
    - Precomputed cover-weighted child probabilities (`yes_weight`, `no_weight`)
    - Active feature set $U_t$ (the unique features split in this tree)
    - Exact vectorized coalitional value evaluation $f_x^{(t)}(S)$ for all $S \subseteq U_t$.
    """

    def __init__(self, tree_dict: Dict[str, Any], feature_to_idx: Dict[str, int]) -> None:
        self.nodes: Dict[int, TreeNode] = {}
        self.feature_to_idx = feature_to_idx
        self.root_id = int(tree_dict.get("nodeid", 0))
        self.postorder_ids: List[int] = []
        self._parse_recursive(tree_dict)
        self._compute_edge_weights()

        # Identify the unique features actually used in splits within this tree
        active_set = sorted({
            node.split_feature_idx
            for node in self.nodes.values()
            if not node.is_leaf
        })
        self.active_features: List[int] = active_set
        self.local_feat_map: Dict[int, int] = {
            feat_idx: local_bit for local_bit, feat_idx in enumerate(self.active_features)
        }
        self.num_active = len(self.active_features)
        self.expected_value = self._compute_expected_value(self.root_id)

    def _parse_recursive(self, d: Dict[str, Any]) -> None:
        nid = int(d["nodeid"])
        if "leaf" in d:
            self.nodes[nid] = TreeNode(
                node_id=nid,
                is_leaf=True,
                leaf_value=float(d["leaf"]),
                cover=float(d.get("cover", 1.0)),
            )
            self.postorder_ids.append(nid)
            return

        feat_name = str(d["split"])
        feat_idx = self.feature_to_idx[feat_name]
        thresh = float(d.get("split_condition", 0.0))
        yes_id = int(d["yes"])
        no_id = int(d["no"])
        missing_id = int(d.get("missing", yes_id))
        cover = float(d.get("cover", 1.0))

        self.nodes[nid] = TreeNode(
            node_id=nid,
            is_leaf=False,
            split_feature=feat_name,
            split_feature_idx=feat_idx,
            split_threshold=thresh,
            yes_id=yes_id,
            no_id=no_id,
            missing_id=missing_id,
            cover=cover,
        )

        for child_dict in d.get("children", []):
            self._parse_recursive(child_dict)
        self.postorder_ids.append(nid)

    def _compute_edge_weights(self) -> None:
        """Compute training cover proportions `R_yes / (R_yes + R_no)` at each internal node."""
        for node in self.nodes.values():
            if not node.is_leaf:
                yes_c = self.nodes[node.yes_id].cover
                no_c = self.nodes[node.no_id].cover
                tot = yes_c + no_c
                if tot > 0.0:
                    node.yes_weight = yes_c / tot
                    node.no_weight = no_c / tot
                else:
                    node.yes_weight = 0.5
                    node.no_weight = 0.5

    def _compute_expected_value(self, nid: int) -> float:
        r"""Compute expected tree output $E[f_t(X)]$ when $S = \emptyset$."""
        node = self.nodes[nid]
        if node.is_leaf:
            return node.leaf_value
        return (
            node.yes_weight * self._compute_expected_value(node.yes_id)
            + node.no_weight * self._compute_expected_value(node.no_id)
        )

    def predict_leaf(self, x: np.ndarray) -> float:
        """Evaluate exact leaf prediction for feature vector `x` when $S = U_t$."""
        nid = self.root_id
        while not self.nodes[nid].is_leaf:
            node = self.nodes[nid]
            val = x[node.split_feature_idx]
            if np.isnan(val):
                nid = node.missing_id
            elif np.float32(val) < np.float32(node.split_threshold):
                nid = node.yes_id
            else:
                nid = node.no_id
        return self.nodes[nid].leaf_value


class CustomXGBoostTreeSHAPExplainer:
    r"""
    Hand-implemented Exact Coalitional TreeSHAP Explainer for an XGBoost Classifier.

    For each tree $t$ with $k = |U_t|$ active features, precomputes the exact Shapley
    combinatorial weights $\frac{|S|!(k - |S| - 1)!}{k!}$ over all $2^k$ coalitions
    $S \subseteq U_t$, evaluates the bottom-up conditional expectation $f_x^{(t)}(S)$
    across all $2^k$ coalitions in vectorized NumPy, and sums exact marginal contributions
    across the entire 250-tree ensemble.
    """

    def __init__(self, xgb_model: xgb.XGBClassifier, feature_names: List[str]) -> None:
        self.xgb_model = xgb_model
        self.booster = xgb_model.get_booster()
        # If model has saved feature names, preserve booster's original training order
        if self.booster.feature_names is not None:
            self.feature_names = list(self.booster.feature_names)
        else:
            self.feature_names = list(feature_names)
        self.num_features = len(self.feature_names)
        self.feature_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}

        tree_dumps = self.booster.get_dump(dump_format="json", with_stats=True)
        self.trees: List[ParsedXGBTree] = [
            ParsedXGBTree(json.loads(td), self.feature_to_idx) for td in tree_dumps
        ]

        # Precompute bitmask lookup tables by number of active features k
        self._coalition_cache: Dict[int, Dict[str, Any]] = {}
        for tree in self.trees:
            k = tree.num_active
            if k > 0 and k not in self._coalition_cache:
                self._coalition_cache[k] = self._build_coalition_tables(k)

        # Determine XGBoost base_score margin shift
        zero_df = pd.DataFrame(np.zeros((1, self.num_features)), columns=self.feature_names)
        raw_margin_zero = float(self.booster.predict(xgb.DMatrix(zero_df), output_margin=True)[0])
        tree_sum_zero = sum(t.predict_leaf(np.zeros(self.num_features)) for t in self.trees)
        self.base_margin_offset = raw_margin_zero - tree_sum_zero

        # Expected value (phi_0) in log-odds space = base_margin_offset + sum of tree expected values
        self.expected_value = self.base_margin_offset + sum(t.expected_value for t in self.trees)

    @staticmethod
    def _build_coalition_tables(k: int) -> Dict[str, Any]:
        """
        Precompute bitmask arrays and exact Shapley factorial weights
        $w(|S|) = \frac{|S|!(k - |S| - 1)!}{k!}$ for a tree with $k$ active features.
        """
        num_masks = 1 << k
        masks = np.arange(num_masks, dtype=np.int32)
        popcounts = np.zeros(num_masks, dtype=np.int32)
        for bit in range(k):
            popcounts += (masks >> bit) & 1

        k_fact = float(math.factorial(k))
        shap_weights_by_size = np.array(
            [
                (math.factorial(s) * math.factorial(k - s - 1)) / k_fact
                if s < k
                else 0.0
                for s in range(k + 1)
            ],
            dtype=np.float64,
        )

        # For each local feature bit `b` in 0..k-1:
        # Identify all masks S where bit `b` is 0, the partner mask S U {b}, and their weights
        bit_tables = []
        for b in range(k):
            bit_flag = 1 << b
            s_without_b = masks[(masks & bit_flag) == 0]
            s_with_b = s_without_b | bit_flag
            weights = shap_weights_by_size[popcounts[s_without_b]]
            bit_tables.append((s_without_b, s_with_b, weights))

        # Also store boolean membership array `(k, 2^k)` indicating whether local bit `b` is in mask `m`
        in_coalition = np.array(
            [((masks >> b) & 1).astype(bool) for b in range(k)],
            dtype=bool,
        )
        return {
            "num_masks": num_masks,
            "in_coalition": in_coalition,
            "bit_tables": bit_tables,
        }

    def _explain_single_tree(self, tree: ParsedXGBTree, x: np.ndarray) -> np.ndarray:
        r"""
        Compute exact Shapley values $\phi_1^{(t)}, \dots, \phi_M^{(t)}$ for tree `t`
        on observation `x` via bottom-up evaluation over all $2^k$ active coalitions.
        """
        phi_tree = np.zeros(self.num_features, dtype=np.float64)
        k = tree.num_active
        if k == 0:
            return phi_tree

        tables = self._coalition_cache[k]
        num_masks = tables["num_masks"]
        in_coalition = tables["in_coalition"]

        # Bottom-up evaluation of conditional expectation V[node_id, mask] for all 2^k coalitions
        node_vals: Dict[int, np.ndarray] = {}
        for nid in tree.postorder_ids:
            node = tree.nodes[nid]
            if node.is_leaf:
                node_vals[nid] = np.full(num_masks, node.leaf_value, dtype=np.float64)
            else:
                local_b = tree.local_feat_map[node.split_feature_idx]
                val = x[node.split_feature_idx]
                if np.isnan(val):
                    hot_id = node.missing_id
                elif np.float32(val) < np.float32(node.split_threshold):
                    hot_id = node.yes_id
                else:
                    hot_id = node.no_id

                v_hot = node_vals[hot_id]
                v_avg = (
                    node.yes_weight * node_vals[node.yes_id]
                    + node.no_weight * node_vals[node.no_id]
                )
                # When local_b is in coalition S, follow hot child; otherwise take cover-weighted average
                node_vals[nid] = np.where(in_coalition[local_b], v_hot, v_avg)

        f_coalitions = node_vals[tree.root_id]

        # Compute exact Shapley value for each active feature in this tree
        for local_b, global_feat_idx in enumerate(tree.active_features):
            s_without_b, s_with_b, weights = tables["bit_tables"][local_b]
            marginal_diffs = f_coalitions[s_with_b] - f_coalitions[s_without_b]
            phi_tree[global_feat_idx] = float(np.dot(weights, marginal_diffs))

        return phi_tree

    def explain_instance(self, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
        r"""
        Compute exact custom TreeSHAP values for a single observation `x` (1D NumPy array).

        Returns:
            shap_values: 1D array of shape `(num_features,)` in log-odds space
            expected_value: scalar $\phi_0$
            pred_margin: scalar $f(x)$ (exact raw log-odds prediction)
        """
        x_arr = np.asarray(x, dtype=np.float64)
        total_phi = np.zeros(self.num_features, dtype=np.float64)
        tree_leaf_sum = 0.0
        for tree in self.trees:
            total_phi += self._explain_single_tree(tree, x_arr)
            tree_leaf_sum += tree.predict_leaf(x_arr)

        pred_margin = self.base_margin_offset + tree_leaf_sum
        return total_phi, self.expected_value, pred_margin

    def explain_batch(self, X: pd.DataFrame | np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Compute custom TreeSHAP values for a batch of observations `X`.
        """
        X_arr = X.to_numpy(dtype=np.float64) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float64)
        n = len(X_arr)
        shap_matrix = np.zeros((n, self.num_features), dtype=np.float64)
        margins = np.zeros(n, dtype=np.float64)

        for idx in range(n):
            phi, _, m = self.explain_instance(X_arr[idx])
            shap_matrix[idx] = phi
            margins[idx] = m

        return shap_matrix, self.expected_value, margins


def generate_custom_shap_plots(
    feature_names: List[str],
    X_sample: pd.DataFrame,
    shap_matrix: np.ndarray,
    case_studies: List[Dict[str, Any]],
    figures_dir: Path,
) -> Dict[str, str]:
    """
    Generate visual plots for our Hand-Implemented XGBoost SHAP:
    1. Global Feature Attribution Summary (Mean |SHAP| + directional impact).
    2. Local Waterfall Explanations for flagged fraudulent transactions (`isFraud = 1`).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------------------
    # Figure 1: Global Mean |SHAP| + Directional Beeswarm-style Strip Plot
    # -------------------------------------------------------------------------
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    order = np.argsort(mean_abs_shap)
    ordered_names = [feature_names[i] for i in order]
    ordered_vals = mean_abs_shap[order]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=140)

    bars = axes[0].barh(ordered_names, ordered_vals, color="#1f78b4", edgecolor="#111111", alpha=0.9)
    axes[0].set_title("Hand-Implemented XGBoost SHAP: Global Mean |φ_i|", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP Value| (Log-Odds Impact)")
    for b, val in zip(bars, ordered_vals):
        axes[0].text(val + 0.02, b.get_y() + b.get_height() / 2.0, f"{val:.3f}", va="center", fontsize=8.5)
    axes[0].set_xlim(0, max(ordered_vals) * 1.18)

    # Panel B: Directional SHAP distribution colored by normalized feature value
    rng = np.random.default_rng(42)
    for pos_idx, feat_idx in enumerate(order):
        shaps = shap_matrix[:, feat_idx]
        fvals = X_sample.iloc[:, feat_idx].to_numpy(dtype=float)
        fmin, fmax = np.percentile(fvals, 5), np.percentile(fvals, 95)
        norm_f = np.clip((fvals - fmin) / max(fmax - fmin, 1e-9), 0.0, 1.0)
        jitter = rng.uniform(-0.22, 0.22, size=len(shaps))
        sc = axes[1].scatter(
            shaps,
            np.full_like(shaps, pos_idx) + jitter,
            c=norm_f,
            cmap="coolwarm",
            s=14,
            alpha=0.75,
        )
    axes[1].axvline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    axes[1].set_yticks(range(len(ordered_names)))
    axes[1].set_yticklabels(ordered_names)
    axes[1].set_title("Hand-Implemented XGBoost SHAP: Directional Feature Impact", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("SHAP Value φ_i (Log-Odds Contribution to Fraud)")
    cbar = plt.colorbar(sc, ax=axes[1], pad=0.02)
    cbar.set_label("Normalized Feature Value (Low -> High)")

    plt.tight_layout()
    global_path = figures_dir / "stage4_custom_xgb_shap_global.png"
    fig.savefig(global_path, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Local Waterfall Explanations for Flagged Fraud Case Studies
    # -------------------------------------------------------------------------
    n_cases = min(2, len(case_studies))
    fig, axes = plt.subplots(1, n_cases, figsize=(13.8, 5.4), dpi=140)
    if n_cases == 1:
        axes = [axes]

    for idx in range(n_cases):
        case = case_studies[idx]
        ax = axes[idx]
        contribs = case["shap_contributions"]
        sorted_items = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8][::-1]
        labels = [
            f"{k} = {case['feature_values'][k]:,.2f}"
            for k, _ in sorted_items
        ]
        vals = [v for _, v in sorted_items]
        colors = ["#e31a1c" if v > 0 else "#1f78b4" for v in vals]

        bars = ax.barh(labels, vals, color=colors, edgecolor="#222222", alpha=0.9)
        ax.axvline(0.0, color="#222222", linewidth=1.1)
        ax.set_title(
            f"Case #{idx+1} (Test Row #{case['test_row_index']}): Flagged Fraud\n"
            f"P(Fraud)={case['predicted_probability']:.4f} | f(x)={case['predicted_log_odds']:.2f} | φ_0={case['base_value_phi0']:.2f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xlabel("Hand-Implemented SHAP Contribution φ_i (Log-Odds)")
        for b, v in zip(bars, vals):
            offset = 0.08 if v >= 0 else -0.08
            ha = "left" if v >= 0 else "right"
            ax.text(
                v + offset,
                b.get_y() + b.get_height() / 2.0,
                f"{v:+.2f}",
                va="center",
                ha=ha,
                fontsize=8.5,
                fontweight="bold",
            )
        max_abs = max(abs(v) for v in vals) if vals else 1.0
        ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)

    plt.tight_layout()
    cases_path = figures_dir / "stage4_custom_xgb_shap_cases.png"
    fig.savefig(cases_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "custom_xgb_shap_global_plot": str(global_path),
        "custom_xgb_shap_cases_plot": str(cases_path),
    }


def run_stage4_custom(config_path: str | Path = "configs/paysim_config.yaml") -> Dict[str, Any]:
    """
    Execute Stage 4A: Run our hand-implemented TreeSHAP algorithm on the trained
    XGBoost classifier, verify the Efficiency Axiom, and explain specific flagged
    fraudulent transactions.
    """
    t0 = time.time()
    config = load_config(config_path)
    modeling_dir = Path(config["paths"]["modeling_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])

    print("[Stage 4A - Custom SHAP] Loading test split and trained XGBoost model...")
    X_test = pd.read_parquet(modeling_dir / "X_test.parquet")
    y_test = pd.read_parquet(modeling_dir / "y_test.parquet")["isFraud"].to_numpy(dtype=int)
    preds_df = pd.read_parquet(models_dir / "test_predictions.parquet")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(models_dir / "xgboost_model.json")

    feature_names = list(X_test.columns)
    print("[Stage 4A - Custom SHAP] Parsing XGBoost JSON trees into CustomXGBoostTreeSHAPExplainer...")
    explainer = CustomXGBoostTreeSHAPExplainer(xgb_model, feature_names)
    max_active = max(t.num_active for t in explainer.trees)
    print(
        f"[Stage 4A - Custom SHAP] Parsed {len(explainer.trees)} trees (max active features per tree = {max_active}). "
        f"Expected Base Value (phi_0) = {explainer.expected_value:.5f}"
    )

    exp_cfg = config.get("explainability", {})
    n_global = int(exp_cfg.get("num_test_samples_global", 300))
    seed = int(exp_cfg.get("random_seed", 42))

    fraud_indices = np.where(y_test == 1)[0]
    legit_indices = np.where(y_test == 0)[0]
    rng = np.random.default_rng(seed)

    sample_fraud_idx = rng.choice(fraud_indices, size=min(n_global // 2, len(fraud_indices)), replace=False)
    sample_legit_idx = rng.choice(legit_indices, size=min(n_global - len(sample_fraud_idx), len(legit_indices)), replace=False)
    global_sample_idx = np.concatenate([sample_fraud_idx, sample_legit_idx])

    X_global_sample = X_test.iloc[global_sample_idx].reset_index(drop=True)
    print(f"[Stage 4A - Custom SHAP] Computing exact coalitional TreeSHAP on {len(X_global_sample)} test transactions...")
    shap_matrix, phi0, margins = explainer.explain_batch(X_global_sample)

    # Verify Efficiency (Local Accuracy) Property: |phi_0 + sum(phi_i) - f(x)| < 1e-10
    reconstructed_margins = phi0 + shap_matrix.sum(axis=1)
    max_efficiency_error = float(np.max(np.abs(reconstructed_margins - margins)))
    mean_efficiency_error = float(np.mean(np.abs(reconstructed_margins - margins)))
    print(
        f"[Stage 4A - Custom SHAP] Mathematical Efficiency Verification: "
        f"Max |phi_0 + sum(phi_i) - f(x)| = {max_efficiency_error:.3e}"
    )

    # Select distinct flagged fraudulent transactions (`isFraud == 1` & flagged by both models) for Case Studies
    both_flagged_mask = (
        (y_test == 1)
        & (preds_df["xgb_prob"].to_numpy() >= 0.80)
        & (preds_df["secondary_prob"].to_numpy() >= 0.80)
    )
    candidate_indices = np.where(both_flagged_mask)[0]
    n_cases = int(exp_cfg.get("num_case_studies", 5))
    selected_case_indices = candidate_indices[:n_cases].tolist()

    case_studies: List[Dict[str, Any]] = []
    for row_idx in selected_case_indices:
        x_row = X_test.iloc[row_idx].to_numpy(dtype=float)
        phi_vec, base_val, pred_margin = explainer.explain_instance(x_row)
        pred_prob = float(1.0 / (1.0 + math.exp(-pred_margin)))
        sum_phi_plus_base = float(base_val + np.sum(phi_vec))
        eff_err = abs(sum_phi_plus_base - pred_margin)

        feat_vals = {col: round(float(X_test.iloc[row_idx][col]), 4) for col in feature_names}
        shap_contribs = {col: round(float(phi_vec[i]), 5) for i, col in enumerate(feature_names)}
        top_drivers = sorted(shap_contribs.items(), key=lambda kv: kv[1], reverse=True)[:3]

        case_studies.append({
            "test_row_index": int(row_idx),
            "true_label": int(y_test[row_idx]),
            "predicted_probability": round(pred_prob, 5),
            "predicted_log_odds": round(float(pred_margin), 5),
            "base_value_phi0": round(float(base_val), 5),
            "reconstructed_sum_phi0_plus_phi": round(sum_phi_plus_base, 5),
            "efficiency_axiom_abs_error": float(f"{eff_err:.3e}"),
            "feature_values": feat_vals,
            "shap_contributions": shap_contribs,
            "top_positive_fraud_drivers": [
                {"feature": k, "shap_log_odds": v, "feature_value": feat_vals[k]}
                for k, v in top_drivers
            ],
        })

    mean_abs_by_feat = {
        col: round(float(np.abs(shap_matrix[:, i]).mean()), 5)
        for i, col in enumerate(feature_names)
    }
    mean_abs_sorted = dict(
        sorted(mean_abs_by_feat.items(), key=lambda kv: kv[1], reverse=True)
    )

    plot_paths = generate_custom_shap_plots(
        feature_names=feature_names,
        X_sample=X_global_sample,
        shap_matrix=shap_matrix,
        case_studies=case_studies,
        figures_dir=figures_dir,
    )

    results = {
        "algorithm": "Hand-Implemented Exact Coalitional TreeSHAP for XGBoost",
        "num_trees_parsed": len(explainer.trees),
        "max_active_features_per_tree": max_active,
        "base_value_phi0_log_odds": round(float(phi0), 5),
        "global_sample_size": int(len(X_global_sample)),
        "mathematical_verification": {
            "axiom": "Local Accuracy / Efficiency: phi_0 + sum(phi_i) == f(x)",
            "max_abs_error": float(f"{max_efficiency_error:.3e}"),
            "mean_abs_error": float(f"{mean_efficiency_error:.3e}"),
            "verified_exact": bool(max_efficiency_error < 1e-6),
        },
        "global_mean_abs_shap": mean_abs_sorted,
        "selected_case_study_indices": selected_case_indices,
        "global_sample_indices": global_sample_idx.tolist(),
        "case_studies": case_studies,
        "plots": plot_paths,
        "execution_time_sec": round(time.time() - t0, 2),
    }

    out_path = models_dir / "custom_xgboost_shap.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[Stage 4A - Custom SHAP] Complete in {results['execution_time_sec']}s -> saved {out_path}")
    return results


if __name__ == "__main__":
    run_stage4_custom()
