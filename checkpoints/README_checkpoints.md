# Model Checkpoints

## Anonymized Access

Pre-trained TRIAGE components (LinearSVC probes + LoRA adapters) are available
at an anonymized location:

> **[ANONYMIZED — available to reviewers upon request via the submission system]**

Full public release on HuggingFace upon acceptance.

---

## What's in the Checkpoint

| File | Description |
|------|-------------|
| `probe_L24.pkl` | LinearSVC probe trained at Layer 24 (PCA-256) |
| `probe_L26.pkl` | LinearSVC probe trained at Layer 26 |
| `lora_P_adapter/` | LoRA adapter weights for Perception-route |
| `lora_K_adapter/` | LoRA adapter weights for Knowledge-route |
| `lora_R_adapter/` | LoRA adapter weights for Reasoning-route |
| `pca_256.pkl` | Fitted PCA transformer (256 components) |

---

## Hardware Note

All checkpoints are compatible with a single NVIDIA RTX 3090 (24GB)
using 4-bit NF4 quantization (bitsandbytes).
