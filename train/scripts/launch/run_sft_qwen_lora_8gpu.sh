#!/usr/bin/env bash
# 8-GPU LoRA SFT for Qwen2.5-Instruct with torchrun.
# Run from repository root:
#   bash train/scripts/launch/run_sft_qwen_lora_8gpu.sh
# Extra CLI args are forwarded to the Python script.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

torchrun \
  --standalone \
  --nproc_per_node=8 \
  train/scripts/sft/train_sft_qwen_lora.py \
  "$@"
