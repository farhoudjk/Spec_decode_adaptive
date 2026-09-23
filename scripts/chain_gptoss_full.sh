#!/bin/bash
set -e
cd "$(dirname "$0")/.."
: "${CUDA_HOME:?set CUDA_HOME (venv nvidia/cuXX path)}"
: "${GPTOSS_DRAFT_PATH:?set GPTOSS_DRAFT_PATH to the patched draft dir}"
LOG=results_gpu_sweep/gptoss_chain.log

echo "[chain] $(date) running 4x3 grid..." | tee -a "$LOG"
scripts/run_gptoss_4x3.sh 2>&1 | tee -a "$LOG"

echo "[chain] $(date) running B0/B1/B2/oracle analysis..." | tee -a "$LOG"
python3 scripts/compute_gptoss_b0_b1_b2.py 2>&1 | tee -a "$LOG"

echo "[chain] $(date) starting adaptive baseline sweep..." | tee -a "$LOG"
scripts/run_gptoss_adaptive_4x3.sh 2>&1 | tee -a "$LOG"

echo "[chain] $(date) ALL DONE" | tee -a "$LOG"
