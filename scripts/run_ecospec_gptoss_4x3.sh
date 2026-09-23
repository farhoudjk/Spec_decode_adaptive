#!/bin/bash
set -e
cd "$(dirname "$0")/.."
: "${CUDA_HOME:?set CUDA_HOME (venv nvidia/cuXX path)}"
: "${GPTOSS_DRAFT_PATH:?set GPTOSS_DRAFT_PATH to the patched draft dir}"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="${HF_HOME:-/mnt/data/hf_cache}"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
OUT=results_gpu_sweep/ecospec_4x3
mkdir -p "$OUT/logs"
for B in 8 16 24 32; do
  for R in 0.5 1 2; do
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
    sleep 4
    echo "===== ECOSPEC-COLLECT B=$B rate=$R =====" | tee -a "$OUT/logs/run.log"
    python3 scripts/collect_ecospec_data.py \
      --model-path openai/gpt-oss-20b \
      --draft-path "$GPTOSS_DRAFT_PATH" \
      --topk-size 4 --steps 2 --eagle-topk 4 \
      --context-length 2048 --B "$B" --rate "$R" --n-requests 30 \
      --attention-backend triton --moe-runner-backend triton \
      --mem-fraction-static 0.94 --cuda-home "$CUDA_HOME" --port 30981 \
      --out "$OUT/B${B}_r${R}" 2>&1 | tee -a "$OUT/logs/run.log"
  done
done
echo ECOSPEC_4X3_DONE | tee -a "$OUT/logs/run.log"
python3 scripts/summarize_ecospec_4x3.py --root "$OUT" --gamma 3
