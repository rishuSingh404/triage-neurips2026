# TriageBench: P/K/R Failure Annotations for Medical VLMs

## Overview

This directory contains the TriageBench annotation layer — the new contribution
of this paper. TriageBench labels 4,755 verified failures from three public
medical VQA datasets with a P/K/R (Perception / Knowledge / Reasoning) etiology.

**Full dataset:** N=4,755 failure samples (anonymized; full release upon acceptance under CC BY 4.0).
**This directory:** 100 stratified samples for reviewer inspection.

---

## Source Datasets

The underlying images and QA pairs are from publicly available datasets:

| Dataset | Domain | License | URL |
|---------|--------|---------|-----|
| VQA-RAD | Radiology | Public domain | https://osf.io/89kps/ |
| SLAKE | Bilingual Medical | CC BY 4.0 | https://www.med-vqa.com/slake/ |
| PathVQA | Histopathology | MIT License | https://github.com/UCSD-AI4H/PathVQA |

We do **not** redistribute the original images or QA pairs.
The P/K/R failure-type labels are our sole contribution.

---

## Sample Distribution (this directory)

| Failure Type | Count | % |
|-------------|-------|---|
| Perception (P) | 80 | 80% |
| Knowledge (K) | 10 | 10% |
| Reasoning (R) | 10 | 10% |

> Note: The P-dominance reflects the natural failure distribution of base VLMs
> on medical VQA (before targeted K/R mining). The full TriageBench dataset
> uses targeted mining to reach 1,000+ samples per K and R etiology.

---

## Files

- `triageBench_sample_100.jsonl` — 100 stratified annotation samples (JSONL, one record per line)
- `annotation_schema.json` — Full field definitions and type constraints

---

## Annotation Protocol

The P/K/R label for each sample is assigned by a tri-model consensus pipeline:

1. **Llama-3.2-90B-Vision** — Primary judge (vision-capable)
2. **Gemini-2.5-Pro** — Secondary judge
3. **GPT-4o** — Tiebreaker when (1) and (2) disagree

The prompt implements a sequential decision tree (see Supplementary A):
- Gate 1: Can the model describe the visual finding? NO → Perception (P)
- Gate 2: Is the medical terminology correct? NO → Knowledge (K)
- Gate 3: Otherwise → Reasoning (R)

Inter-annotator agreement with board-certified radiologists:
Cohen's κ = 0.75 (Substantial Agreement, N=1,200 sampled).

---

## License

P/K/R annotation labels: **CC BY 4.0** (upon acceptance).
Source images and QA pairs retain their original licenses (see table above).
