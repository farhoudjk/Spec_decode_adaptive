#!/bin/bash
set -e
cd "$(dirname "$0")/.."
: "${CUDA_HOME:?set CUDA_HOME (venv nvidia/cuXX path)}"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/data/hf_cache}"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

python3 scripts/sweep_sglang_verify_footprint.py \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --draft-path lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex \
  --num-layers 48 --topk-size 8 \
  --steps 1 2 3 4 6 8 --topks 1 2 4 8 \
  --Bs 8 16 24 32 --rates 2 4 12 \
  --duration 60 --context-length 2048 --rtype code \
  --attention-backend triton --port 30100 \
  --out results_gpu_sweep/qwen_4x3
