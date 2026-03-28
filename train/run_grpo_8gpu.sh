#!/usr/bin/env bash
# 8-GPU GRPO + DeepSpeed ZeRO-3 + vLLM (colocate, default in train_grpo_qwen.py).
# Run from repository root:
#   bash train/run_grpo_8gpu.sh
# Optional extra args are forwarded to the Python script.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

accelerate launch \
  --config_file train/accelerate_deepspeed_zero3.yaml \
  train/train_grpo_qwen.py \
  "$@"
