#!/bin/bash
set -e
cd "$(dirname "$0")/.."
: "${CUDA_HOME:?set CUDA_HOME (venv nvidia/cuXX path)}"
: "${GPTOSS_DRAFT_PATH:?set GPTOSS_DRAFT_PATH to the patched draft dir (see rebuild_env_gptoss.sh)}"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/data/hf_cache}"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
OUT=results_gpu_sweep/gptoss_4x3
mkdir -p "$OUT/logs"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
sleep 6
python3 scripts/sweep_sglang_verify_footprint.py \
  --model-path openai/gpt-oss-20b \
  --draft-path "$GPTOSS_DRAFT_PATH" \
  --num-layers 24 --topk-size 4 \
  --steps 1 2 3 4 6 8 --topks 1 2 4 8 --Bs 8 16 24 32 --rates 0.5 1 2 \
  --duration 40 --context-length 2048 --rtype code \
  --attention-backend triton --moe-runner-backend triton \
  --mem-fraction-static 0.94 --request-timeout 300 --port 30930 \
  --out "$OUT" 2>&1 | tee -a "$OUT/logs/run.log"
echo GRID_DONE | tee -a "$OUT/logs/run.log"
