#!/usr/bin/env python3
"""
train_probe.py — TRIAGE Linear Probe Training

Trains binary LinearSVC probes on prefill hidden states to detect
P/K/R failure etiologies before generation.

Usage:
    python src/probes/train_probe.py \
        --hidden_states path/to/hidden_states.pt \
        --labels path/to/labels.json \
        --output_dir outputs/probes/

Architecture:
    - LinearSVC (C=0.1, dual=True) trained on PCA-256 features
    - Stratified 5-fold cross-validation
    - Per-layer AUROC to identify the phase transition layer (peaks at L24-L26)
"""
import argparse
import json
import os
import torch
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm


def load_hidden_states(path: str) -> dict:
    """Load extracted hidden states from disk.

    Returns:
        dict with keys: 'states' (tensor: N x L x D), 'labels' (list of str)
    """
    return torch.load(path, map_location="cpu")


def compute_layer_aurocs(states: np.ndarray, labels: np.ndarray, n_layers: int) -> list:
    """
    For each layer, train a LinearSVC probe and compute macro-AUROC via 5-fold CV.

    Returns list of AUROC values, one per layer.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pca = PCA(n_components=256, random_state=42)
    layer_aurocs = []

    for layer_idx in tqdm(range(n_layers), desc="Probing layers"):
        X = states[:, layer_idx, :]          # (N, D)
        X_pca = pca.fit_transform(X)

        fold_aurocs = []
        for train_idx, val_idx in skf.split(X_pca, labels):
            X_tr, X_val = X_pca[train_idx], X_pca[val_idx]
            y_tr, y_val = labels[train_idx], labels[val_idx]

            clf = LinearSVC(C=0.1, dual=True, max_iter=2000)
            clf.fit(X_tr, y_tr)

            # Compute decision scores for one-vs-rest AUROC
            scores = clf.decision_function(X_val)
            classes = clf.classes_
            y_bin = label_binarize(y_val, classes=classes)
            try:
                auroc = roc_auc_score(y_bin, scores, multi_class="ovr", average="macro")
            except ValueError:
                auroc = 0.5
            fold_aurocs.append(auroc)

        layer_aurocs.append(float(np.mean(fold_aurocs)))

    return layer_aurocs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_states", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output_dir", default="outputs/probes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/3] Loading hidden states...")
    data = load_hidden_states(args.hidden_states)
    states = data["states"].numpy()      # (N, L, D)
    with open(args.labels) as f:
        labels = np.array(json.load(f))  # ["P", "K", "R", ...]

    n_layers = states.shape[1]
    print(f"     N={len(states)} samples, L={n_layers} layers, D={states.shape[2]}")

    print("[2/3] Running layer-wise probing...")
    aurocs = compute_layer_aurocs(states, labels, n_layers)

    best_layer = int(np.argmax(aurocs))
    print(f"     Phase transition layer: L{best_layer} (AUROC={aurocs[best_layer]:.4f})")

    print("[3/3] Saving results...")
    results = {
        "layer_aurocs": aurocs,
        "best_layer": best_layer,
        "best_auroc": aurocs[best_layer],
    }
    out_path = os.path.join(args.output_dir, "probe_aurocs.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"     Saved to {out_path}")


if __name__ == "__main__":
    main()
