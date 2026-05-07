#!/bin/bash
# reproduce_table3.sh — Table 3: Per-Etiology Correction Breakdown
# Expected: P=57.8%, K=34.2%, R=62.2%
set -e
echo "================================================================"
echo " TRIAGE — Table 3: Per-Etiology Breakdown"
echo "================================================================"
python src/routing/triage_pipeline.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --probe_path outputs/probes/probe_L24.pkl \
    --dataset vqa-rad --split test \
    --output results/table3/triage_etiology.json
echo "Done! Expected per-type correction: P=57.8%, K=34.2%, R=62.2%"
