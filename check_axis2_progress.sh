#!/bin/bash
echo "=== last 5 log lines ==="
tail -5 results_gpu_sweep/axis2_sweep.log
echo
echo "=== cells completed ==="
python3 -c "
import json
d = json.load(open('results_gpu_sweep/axis2/grid.json'))
ok = sum(1 for v in d.values() if 'error' not in v)
err = sum(1 for v in d.values() if 'error' in v)
print(f'{ok} completed, {err} failed, out of 96 total')
"
