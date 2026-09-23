#!/bin/bash
set -e
mountpoint -q /mnt/data || { echo "FATAL: /mnt/data not mounted (see header)"; exit 1; }
echo "[1/4] venv"
python3 -m venv /mnt/data/venv
/mnt/data/venv/bin/pip install -q --upgrade pip wheel
echo "[2/4] sglang + ninja (long)"
TMPDIR=/mnt/data/tmp_pip /mnt/data/venv/bin/pip install --cache-dir /mnt/data/pip_cache "sglang[all]" ninja
echo "[3/4] hf cache symlinks (draft + datasets already on root disk)"
mkdir -p /mnt/data/hf_cache/hub
for d in models--lmsys--SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex \
         datasets--openai--openai_humaneval datasets--rajpurkar--squad datasets--abisee--cnn_dailymail; do
  ln -sfn /home/ubuntu/.cache/huggingface/hub/$d /mnt/data/hf_cache/hub/$d
done
echo "[4/4] Qwen FP8 target (~31GB)"
HF_HOME=/mnt/data/hf_cache /mnt/data/venv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-30B-A3B-Instruct-2507-FP8', max_workers=8))"
echo "REBUILD_DONE"
