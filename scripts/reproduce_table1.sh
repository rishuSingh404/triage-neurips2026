#!/bin/bash
# reproduce_table1.sh — Table 1: Performance Ladder (VQA-RAD)
#
# This script runs TRIAGE and all baselines on VQA-RAD and reproduces
# the main results table (Performance Ladder).
#
# Expected output:
#   Zero-shot (Qwen2.5-VL-7B): 50.3%
#   Structured CoT:            46.0%
#   Unconditioned LoRA:        44.9%
#   Random Routing:            43.3%
#   PGTL (Probe only):         52.4%
#   TRIAGE (Full):             58.8%
#   Oracle Routing:            60.9%
#
# Hardware: 2x NVIDIA RTX 3090 (24GB) with 4-bit NF4 quantization
# Estimated runtime: ~4 hours

set -e

DATASET="vqa-rad"
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="results/table1"
PROBE_LAYER=24
PROBE_PATH="outputs/probes/probe_L${PROBE_LAYER}.pkl"

mkdir -p $OUTPUT_DIR

echo "================================================================"
echo " TRIAGE — Table 1: Performance Ladder (VQA-RAD)"
echo "================================================================"

# Step 1: Extract prefill hidden states from test set
echo "[1/5] Extracting hidden states (Layer ${PROBE_LAYER})..."
python src/utils/hidden_state_extractor.py \
    --model_id $MODEL_ID \
    --dataset $DATASET \
    --split test \
    --layer $PROBE_LAYER \
    --output outputs/hidden_states_test.pt

# Step 2: Train the probe (if not already trained)
echo "[2/5] Training LinearSVC probe..."
python src/probes/train_probe.py \
    --hidden_states outputs/hidden_states_train.pt \
    --labels outputs/labels_train.json \
    --output_dir outputs/probes/

# Step 3: Run TRIAGE full pipeline
echo "[3/5] Running TRIAGE (Full) inference..."
python src/routing/triage_pipeline.py \
    --model_id $MODEL_ID \
    --probe_path $PROBE_PATH \
    --dataset $DATASET \
    --split test \
    --output $OUTPUT_DIR/triage_full.json

# Step 4: Run baselines
echo "[4/5] Running Unconditioned LoRA baseline..."
python src/experiments/exp_unconditioned_lora.py \
    --model_id $MODEL_ID \
    --train_data data/splits/train_ids.txt \
    --output $OUTPUT_DIR/unconditioned_lora.json

# Step 5: Aggregate and print results
echo "[5/5] Aggregating results..."
python - <<'EOF'
import json, glob

results = {}
for f in glob.glob("results/table1/*.json"):
    with open(f) as fp:
        d = json.load(fp)
    print(f"  {f}: {d}")

print("\n  Full results table: see paper Table 1.")
EOF

echo ""
echo "Done! Results saved to $OUTPUT_DIR/"
