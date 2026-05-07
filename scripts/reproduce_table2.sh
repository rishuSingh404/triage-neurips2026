#!/bin/bash
# reproduce_table2.sh — Table 2: One-Way Geometric Barrier
# Expected: Forward rescue 25.2%, Backward corrupt 87.2%, Noise control 39.0%
set -e
echo "================================================================"
echo " TRIAGE — Table 2: One-Way Geometric Barrier"
echo "================================================================"
python src/experiments/exp_centroid_patching.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --hidden_states outputs/hidden_states_test.pt \
    --labels outputs/labels_test.json \
    --n_samples 250 \
    --patch_layer 27 \
    --output results/table2/barrier_qwen.json
echo "Done! Expected: Fwd=25.2%, Bwd=87.2%, Noise=39.0%"
