#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 train/scripts/grpo/train_grpo_quotes_summary_multi.py \
  --train_file dataset/quotes_summary_5d_2026-01-01_to_2026-03-01.csv \
  --eval_file dataset/quotes_summary_5d_2026-03-01_to_2026-04-01.csv \
  --output_dir /nfs/hanpeng/huggingface/models/qwen2_5_grpo_quotes_summary_multi \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_generations 8 \
  --max_prompt_length 6144 \
  --max_completion_length 768 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_steps 500 \
  --save_total_limit 3 \
  --report_to none \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.35 \
  "$@"
