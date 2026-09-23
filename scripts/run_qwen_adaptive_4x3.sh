#!/bin/bash
set -e
cd "$(dirname "$0")/.."
: "${CUDA_HOME:?set CUDA_HOME (venv nvidia/cuXX path)}"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/data/hf_cache}"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
OUT=results_gpu_sweep/qwen_adaptive_4x3
mkdir -p "$OUT/logs"
PORT=30200
for B in 8 16 24 32; do
  for R in 2 4 12; do
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
    sleep 4
    PORT=$((PORT + 1))
    echo "===== ADAPTIVE B=$B rate=$R =====" | tee -a "$OUT/logs/run.log"
    python3 scripts/sweep_lambda_gap_adaptive.py \
      --batches "$B" --lambdas "$R" --depths 2 3 4 6 8 --topk 1 \
      --num-draft-tokens 9 --duration 60 --port "$PORT" \
      --hf-home "$HF_HOME" --cuda-home "$CUDA_HOME" \
      --out-dir "$OUT" 2>&1 | tee -a "$OUT/logs/run.log"
  done
done
echo ADAPTIVE_4X3_DONE | tee -a "$OUT/logs/run.log"
