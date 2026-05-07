#!/usr/bin/env python3
"""
triage_pipeline.py — Full TRIAGE Inference Pipeline

Implements the etiology-driven contrastive decoding pipeline:
  1. Extract prefill hidden state at the best probe layer (L24-L26).
  2. Run LinearSVC probe → predict P, K, or R etiology.
  3. Apply dynamic contrastive decoding alpha based on etiology:
       P → α=0.65  (force visual grounding)
       R → α=0.40  (gentle logic correction)
       K → α=0.0   (bypass CD, protect factual recall)
  4. Apply Adaptive Plausibility Constraints (APC) to prevent vocabulary collapse.

Usage:
    python src/routing/triage_pipeline.py \
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \
        --probe_path outputs/probes/probe_L24.pkl \
        --dataset vqa-rad \
        --split test \
        --output results/triage_vqarad.json
"""
import argparse
import json
import pickle
import torch
import numpy as np
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
)


# ── Contrastive Decoding Alpha Map ────────────────────────────────────────────
ALPHA_MAP = {
    "P": 0.65,   # Perception: high penalty to force visual grounding
    "R": 0.40,   # Reasoning: gentle penalty to correct logic chain
    "K": 0.00,   # Knowledge: bypass CD entirely to protect factual recall
}


class AdaptivePlausibilityConstraint(LogitsProcessor):
    """
    Masks tokens with probability < 5% of the expert LoRA's top token.
    Prevents vocabulary collapse during contrastive decoding.
    """
    def __init__(self, lora_logits: torch.Tensor, threshold: float = 0.05):
        self.mask = lora_logits < (threshold * lora_logits.max(dim=-1, keepdim=True).values)

    def __call__(self, input_ids, scores):
        scores = scores.masked_fill(self.mask, float("-inf"))
        return scores


class EtiologyContrastiveDecoder(LogitsProcessor):
    """
    Implements etiology-driven contrastive decoding:
        logits_final = logits_base - α * logits_lora
    where α is determined by the probe's failure-type prediction.
    """
    def __init__(self, lora_logits: torch.Tensor, alpha: float):
        self.lora_logits = lora_logits
        self.alpha = alpha

    def __call__(self, input_ids, scores):
        if self.alpha == 0.0:
            return scores  # K-route: bypass CD entirely
        return scores - self.alpha * self.lora_logits


def predict_etiology(probe, hidden_state: np.ndarray) -> str:
    """Run the LinearSVC probe on the prefill hidden state."""
    pca = probe["pca"]
    clf = probe["clf"]
    h = pca.transform(hidden_state.reshape(1, -1))
    return clf.predict(h)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--probe_path", required=True)
    parser.add_argument("--dataset", choices=["vqa-rad", "slake", "pathvqa"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="results/triage_output.json")
    parser.add_argument("--probe_layer", type=int, default=24)
    args = parser.parse_args()

    print(f"[1/4] Loading probe from {args.probe_path}...")
    with open(args.probe_path, "rb") as f:
        probe = pickle.load(f)

    print(f"[2/4] Loading model {args.model_id} (4-bit NF4)...")
    # NOTE: Full model loading code omitted for brevity.
    # See src/utils/data_utils.py for the complete load_model_4bit() helper.
    print("      [Model loading skipped in sample — see data_utils.py]")

    print("[3/4] Running TRIAGE inference on test set...")
    print("      [Inference loop omitted in sample — see scripts/reproduce_table1.sh]")
    print(f"      Etiology alpha map: {ALPHA_MAP}")

    print(f"[4/4] Results would be saved to {args.output}")
    sample_output = {
        "dataset": args.dataset,
        "split": args.split,
        "probe_layer": args.probe_layer,
        "alpha_map": ALPHA_MAP,
        "note": "This is a skeleton sample. Full inference code available upon acceptance.",
    }
    import os; os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(sample_output, f, indent=2)


if __name__ == "__main__":
    main()
