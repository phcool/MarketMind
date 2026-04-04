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
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HOME="${HF_HOME:-/nfs/hanpeng/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "[run_sft] repo_root=$ROOT"
echo "[run_sft] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[run_sft] omp_num_threads=$OMP_NUM_THREADS"
echo "[run_sft] hf_home=$HF_HOME"
echo "[run_sft] hf_datasets_cache=$HF_DATASETS_CACHE"
echo "[run_sft] transformers_cache=$TRANSFORMERS_CACHE"
echo "[run_sft] master_addr=$MASTER_ADDR"
echo "[run_sft] master_port=$MASTER_PORT"

if ! command -v uv >/dev/null 2>&1; then
  echo "[run_sft] error: uv not found in PATH=$PATH" >&2
  exit 127
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "[run_sft] error: .venv is missing. Run: uv sync --extra train" >&2
  exit 1
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
echo "[run_sft] uv_cache_dir=$UV_CACHE_DIR"

uv run --no-sync torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train/scripts/sft/train_sft_qwen_lora.py \
  "$@"
