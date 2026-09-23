#!/bin/bash
set -e
mountpoint -q /mnt/data || { echo "FATAL: /mnt/data not mounted"; exit 1; }
echo "[1/5] venv"
python3 -m venv /mnt/data/venv
/mnt/data/venv/bin/pip install -q --upgrade pip wheel
echo "[2/5] sglang + ninja (long)"
TMPDIR=/mnt/data/tmp_pip /mnt/data/venv/bin/pip install --cache-dir /mnt/data/pip_cache "sglang[all]" ninja
echo "[3/5] hf cache symlinks"
mkdir -p /mnt/data/hf_cache/hub
for d in datasets--openai--openai_humaneval datasets--rajpurkar--squad datasets--abisee--cnn_dailymail; do
  ln -sfn /home/ubuntu/.cache/huggingface/hub/$d /mnt/data/hf_cache/hub/$d
done
echo "[4/5] gpt-oss-20b target (13.8GB) + nebius draft"
HF_HOME=/mnt/data/hf_cache /mnt/data/venv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('openai/gpt-oss-20b', max_workers=8,
      allow_patterns=['*.safetensors','*.json','*.jinja','*.txt']))
print(snapshot_download('nebius/EAGLE3-gpt-oss-20b', max_workers=4))"
echo "[5/5] patch draft architectures -> LlamaForCausalLMEagle3"
SRC=$(ls -d /mnt/data/hf_cache/hub/models--nebius--EAGLE3-gpt-oss-20b/snapshots/*/ | head -1)
DST=/mnt/data/eagle3-gptoss-20b-sglang
mkdir -p $DST
cp -L "$SRC/model.safetensors" $DST/
/mnt/data/venv/bin/python -c "
import json
c=json.load(open('$SRC/config.json'))
c['architectures']=['LlamaForCausalLMEagle3']
json.dump(c,open('$DST/config.json','w'),indent=2)
print('patched ->',c['architectures'])"
echo "REBUILD_DONE"
echo "Set this before running gpt-oss scripts:"
echo "  export GPTOSS_DRAFT_PATH=$DST"
echo "  export CUDA_HOME=/mnt/data/venv/lib/python3.12/site-packages/nvidia/cu13"
