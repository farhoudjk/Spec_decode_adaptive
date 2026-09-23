#!/bin/bash
set -e
cd "$(dirname "$0")/.."
: "${CUDA_HOME:?set CUDA_HOME (venv nvidia/cuXX path)}"
: "${GPTOSS_DRAFT_PATH:?set GPTOSS_DRAFT_PATH to the patched draft dir (see rebuild_env_gptoss.sh)}"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/data/hf_cache}"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
OUT=results_gpu_sweep/gptoss_adaptive_4x3
mkdir -p "$OUT/logs"
PORT=30950
for B in 8 16 24 32; do
  for R in 0.5 1 2; do
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
    sleep 4
    echo "===== ADAPTIVE B=$B rate=$R =====" | tee -a "$OUT/logs/run.log"
    python3 scripts/sweep_lambda_gap_adaptive.py \
      --batches "$B" --lambdas "$R" --depths 1 2 3 4 6 8 --topk 1 \
      --num-draft-tokens 9 --duration 40 --request-timeout 300 \
      --model-path openai/gpt-oss-20b \
      --draft-path "$GPTOSS_DRAFT_PATH" \
      --moe-runner-backend triton --mem-fraction-static 0.94 \
      --cuda-home "$CUDA_HOME" --hf-home "$HF_HOME" \
      --port "$PORT" --out-dir "$OUT" 2>&1 | tee -a "$OUT/logs/run.log"
  done
done
echo ADAPTIVE_4X3_DONE | tee -a "$OUT/logs/run.log"
