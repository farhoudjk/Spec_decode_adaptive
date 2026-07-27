#!/usr/bin/env bash
# Full experiment matrix on real GPU. Run tests/test_patch_contract.py FIRST.
# Usage: CONFIG=configs/rtx4090_24gb.yaml scripts/run_matrix.sh
set -euo pipefail
CONFIG="${CONFIG:-configs/rtx4090_24gb.yaml}"
OUT="${OUT:-results_gpu}"
RATE="${RATE:-8.0}"
DUR="${DUR:-180.0}"
SEEDS="${SEEDS:-0 1 2}"
mkdir -p "$OUT"

# ---- helper: write a variant config with a patched controller block --------
mkvar () { python3 scripts/mkconfig.py "$CONFIG" "$1" > "$OUT/cfg_$2.yaml"; echo "$OUT/cfg_$2.yaml"; }

echo "== contract test (no GPU needed) =="
python3 tests/test_patch_contract.py

echo "== RQ3/ablation arms: the paper's core comparison =="
# baseline = vLLM's shipped open-loop table ; ours = closed-loop
declare -A ARMS=(
  [no_spec]='{"spec":"static","admit":"static","spec_kw":{"gamma":0},"admit_kw":{"max_num_seqs":64}}'
  [static_spec]='{"spec":"static","admit":"static","spec_kw":{"gamma":4},"admit_kw":{"max_num_seqs":64}}'
  [vllm_open_loop]='{"spec":"static-table","admit":"slack","coordination":"naive","admit_kw":{"gain":0.5,"period":4,"init":64}}'
  [ours_closed_loop]='{"spec":"closed-loop","admit":"slack","coordination":"naive","spec_kw":{"gain":0.5,"period":4},"admit_kw":{"gain":0.5,"period":4,"init":64}}'
  [spec_only]='{"spec":"closed-loop","admit":"static","spec_kw":{"gain":0.5,"period":4},"admit_kw":{"max_num_seqs":64}}'
  [admit_only]='{"spec":"static","admit":"slack","spec_kw":{"gamma":4},"admit_kw":{"gain":0.5,"period":4,"init":64}}'
)
for arm in "${!ARMS[@]}"; do
  cfg=$(mkvar "${ARMS[$arm]}" "$arm")
  for s in $SEEDS; do
    echo ">> arm=$arm seed=$s"
    python3 -m specloop_rt.replay --config "$cfg" --trace mixed \
        --rate "$RATE" --duration "$DUR" --seed "$s" --out "$OUT/${arm}_s${s}"
  done
done

echo "== RQ4 coordination on the perturbation trace =="
for coord in naive timescale hysteresis; do
  cfg=$(mkvar "{\"spec\":\"closed-loop\",\"admit\":\"slack\",\"coordination\":\"$coord\",\"spec_kw\":{\"gain\":0.5,\"period\":4},\"admit_kw\":{\"gain\":0.5,\"period\":4,\"init\":64},\"coord_kw\":{}}" "coord_$coord")
  for s in $SEEDS; do
    python3 -m specloop_rt.replay --config "$cfg" --trace step \
        --rate "$RATE" --duration "$DUR" --seed "$s" --out "$OUT/coord_${coord}_s${s}"
  done
done

echo "== analyze =="
python3 scripts/analyze.py "$OUT" --tpot-slo 0.040 --ttft-slo 2.0 > "$OUT/summary.txt"
cat "$OUT/summary.txt"
