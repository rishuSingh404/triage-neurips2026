#!/usr/bin/env python3
"""
train_probe.py — TRIAGE Linear Probe Training (Illustrative Sample)

Trains binary LinearSVC probes on prefill hidden states to detect
P/K/R failure etiologies before generation begins.

This file shows the training configuration and evaluation protocol.
The complete data loading pipeline and training loop will be released
upon paper acceptance.

Key design choices (from paper Section 3.2):
  - Probe type:    LinearSVC (C=0.1, dual=True)
  - Features:      PCA-256 projection of last-token prefill hidden state
  - Validation:    Stratified 5-fold cross-validation
  - Best layer:    L24–L26 (identified via the layer-wise AUROC sweep shown in Figure 2a)
  - Null baseline: 100 independent label shuffles → AUROC 0.505 ± 0.02 (chance)
"""
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


# ── Probe Configuration ───────────────────────────────────────────────────────
PROBE_CONFIG = {
    "classifier":   "LinearSVC",
    "C":            0.1,
    "dual":         True,
    "max_iter":     2000,
    "pca_dim":      256,
    "cv_folds":     5,
    "random_state": 42,
}

# ── Reported AUROC Results (paper Table 2 / Figure 2a) ───────────────────────
REPORTED_AUROCS = {
    "P": {"best_layer": 24, "auroc": 0.787, "ci_95": [0.771, 0.802]},
    "K": {"best_layer": 26, "auroc": 0.803, "ci_95": [0.787, 0.818]},
    "R": {"best_layer": 24, "auroc": 0.857, "ci_95": [0.843, 0.870]},
    "macro_avg": 0.816,
}


def build_probe(pca_dim: int = 256, C: float = 0.1):
    """
    Returns a (pca, clf) tuple ready for fitting.

    Usage:
        pca, clf = build_probe()
        X_pca = pca.fit_transform(hidden_states)   # (N, pca_dim)
        clf.fit(X_pca, labels)
        probe = {"pca": pca, "clf": clf}
    """
    pca = PCA(n_components=pca_dim, random_state=42)
    clf = LinearSVC(C=C, dual=True, max_iter=2000)
    return pca, clf


# ──────────────────────────────────────────────────────────────────────────────
#  NOTE: The following components are redacted pending paper acceptance.
#  Full code will be released under the MIT License upon acceptance.
#
#  Redacted components:
#    1. load_hidden_states()  — Loads pre-extracted .pt files from disk
#    2. compute_layer_aurocs() — Full layer-wise CV sweep (L0–L28)
#    3. save_probe()           — Serialises (pca, clf) to .pkl
#    4. main()                 — CLI entrypoint with argparse
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("TRIAGE Linear Probe — Configuration")
    print(f"  Classifier: {PROBE_CONFIG['classifier']}(C={PROBE_CONFIG['C']})")
    print(f"  PCA dim:    {PROBE_CONFIG['pca_dim']}")
    print(f"  CV folds:   {PROBE_CONFIG['cv_folds']}")
    print()
    print("  Reported AUROCs (from paper):")
    for etype, v in REPORTED_AUROCS.items():
        if isinstance(v, dict):
            print(f"    {etype}: AUROC={v['auroc']} at L{v['best_layer']}  95%CI={v['ci_95']}")
    print(f"    Macro avg: {REPORTED_AUROCS['macro_avg']}")
    print()
    print("  Full training pipeline will be released upon paper acceptance.")
