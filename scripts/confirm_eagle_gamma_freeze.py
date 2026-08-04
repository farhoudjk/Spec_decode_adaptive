"""Confirming test: does EAGLE's num_speculative_tokens actually vary
accept_rate/ITL when set correctly at engine construction (gamma_init per
cell, full rebuild), vs the sweep_predictive.py bug where only spec_kw's
gamma_init (a controller-internal param) was set and cfg["runtime"]
["gamma_init"] -- what _build_engine actually reads to construct
speculative_config -- stayed at whatever the base config had?

Root cause (confirmed by reading vllm source directly, vllm==0.9.2):
  vllm/v1/spec_decode/eagle.py, EagleProposer.__init__:
      self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
  never reassigned after __init__ -- propose() uses this frozen instance
  attribute. specloop_rt's scheduler_patch.py writes a new value into
  vllm_config.speculative_config.num_speculative_tokens every step (trying
  to actuate gamma live), but the EagleProposer already copied the OLD value
  into its own attribute at construction and never looks at the config
  object again. Same freeze exists in vllm/v1/spec_decode/ngram_proposer.py.

This means: any spec controller's live actuation (ClosedLoopSpec, GatedSpec,
HillClimbSpec, DSDESpec) is a no-op against the REAL proposer for both ngram
and EAGLE in vllm 0.9.2 -- gamma_current in telemetry is what the controller
THINKS it set, not what the drafter actually used. The only way to get a
different k is a fresh engine (fresh proposer instance) per k, i.e. exactly
what sweep_roofline_moe.py did correctly (cfg["runtime"]["gamma_init"] = k
per cell) and sweep_predictive.py / sweep_hillclimb_vs_static.py did NOT.

This script runs 3 cells at gamma_init in {1, 4, 8}, static/static
controller (so there's no live-actuation confound at all -- gamma is set
once and never touched), same B/rate/workload as the original Axis-4 EAGLE
config. If accept_rate/ITL vary meaningfully across these 3 cells, that
confirms speculation depth DOES have real leverage on EAGLE (matching the
Mixtral-ngram fit sweep's finding for a different backend+model), and the
original Axis-4 "gamma has no leverage" conclusion was an artifact of the
freeze bug, not a property of EAGLE or dense models.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specloop_rt.analysis import summarize_run

GAMMAS = [1, 4, 8]
B = 16
RATE = 6.0
DURATION = 90.0
OUT = "results_gpu_sweep/axis4_eagle_gamma_confirm"


def main():
    with open("configs/a5000_24gb_llama3_eagle.yaml") as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs(OUT, exist_ok=True)
    grid_path = os.path.join(OUT, "grid.json")
    grid = json.load(open(grid_path)) if os.path.exists(grid_path) else {}

    for gamma in GAMMAS:
        key = f"gamma{gamma}_B{B}_r{RATE}"
        if key in grid and "error" not in grid[key]:
            print(f".. skip {key}", flush=True)
            continue
        cfg = copy.deepcopy(base_cfg)
        cfg["runtime"]["gamma_init"] = gamma          # <-- the fix: actually
        cfg["runtime"]["max_num_seqs_init"] = B        #     rebuild the engine
        cfg["controller"] = {                          #     with this k
            "spec": "static", "admit": "static", "coordination": "naive",
            "spec_kw": {"gamma": gamma},
            "admit_kw": {"max_num_seqs": B},
            "coord_kw": {},
        }
        cfg_path = os.path.join(OUT, f"cfg_{key}.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        run_dir = os.path.join(OUT, key)
        cmd = [sys.executable, "-m", "specloop_rt.replay", "--config", cfg_path,
               "--trace", "homogeneous", "--rtype", "code", "--rate", str(RATE),
               "--duration", str(DURATION), "--seed", "0",
               "--real-corpus", "--out", run_dir]
        print(f">> gamma={gamma} B={B} rate={RATE}", flush=True)
        try:
            subprocess.run(cmd, check=True)
            m = summarize_run(run_dir, tpot_slo=base_cfg["runtime"]["tpot_slo_s"],
                              ttft_slo=base_cfg["runtime"]["ttft_slo_s"])
            grid[key] = {"gamma": gamma, "B": B, "rate": RATE, **m}
        except subprocess.CalledProcessError as e:
            print(f"!! {key} FAILED: {e}", flush=True)
            grid[key] = {"gamma": gamma, "B": B, "rate": RATE, "error": str(e)}
        with open(grid_path, "w") as f:
            json.dump(grid, f, indent=2)

    print("\n=== gamma_init (real, per-cell rebuild) vs accept_rate/ITL ===")
    for gamma in GAMMAS:
        key = f"gamma{gamma}_B{B}_r{RATE}"
        v = grid.get(key, {})
        if "error" in v:
            print(f"gamma={gamma}: FAILED")
            continue
        print(f"gamma={gamma}: tpot_p50={v.get('tpot_p50')}  "
              f"mean_accept_rate={v.get('mean_accept_rate')}  "
              f"mean_gamma={v.get('mean_gamma')}")


if __name__ == "__main__":
    main()
