"""Axis-7 / C1, take 3: FULL verify-batch expert-footprint sweep (proposed
draft tokens, accepted AND rejected). See AXIS7.md#10 for the full trace of
why this exists -- takes 1 and 2 are both documented dead ends/partial
results:

- take 1 (specloop_rt/sglang_patch/moe_expert_hooks.py, TopK.forward patch):
  never fires on real traffic, CUDA graph replay bypasses it (AXIS7.md#2).
- take 2 (scripts/sweep_sglang_expert_footprint.py, native
  --enable-return-routed-experts): works, but is target-model,
  ACCEPTED-TOKEN-ONLY -- confirmed live, and a clean 9-cell GPU grid found
  the resulting distinct-experts/imbalance stats are flat across D and W
  (AXIS7.md#6), which is a real but INCONCLUSIVE result on the design
  brief's actual hypothesis (which is about the rejected-inclusive
  verify-batch footprint, since rejected candidates still cost GEMM
  compute during verify).

THIS SWEEP uses specloop_rt/sglang_patch/verify_batch_expert_hooks.py,
which patches ModelRunner.forward (not TopK.forward) to intercept the
verify-batch's routed_experts BEFORE finalize() narrows it to
accepted-only KV-cache positions. Unlike take 1, this patch point is
naturally CUDA-graph-safe: on_forward_end() (which the patch wraps around,
by wrapping the whole forward() call) runs strictly after graph replay
completes for that step, in eager Python -- see that module's docstring
for the exact code-path trace confirming this.

Needs its own PYTHONPATH + env wiring (like take 1 did, unlike take 2's
client-side-only flag) since this is a real monkeypatch that must install
inside SGLang's spawned scheduler subprocess via sitecustomize.py.

Separate sweep from both sweep_sglang_depth_width.py (axis6) and
sweep_sglang_expert_footprint.py (take 2), for the same reasons take 1's
version gave: unknown sync/latency cost (a per-verify-step .to("cpu") call,
smaller than take 1's per-MoE-layer sync since it's now one copy per
verify step covering all layers at once, but still not free -- not yet
measured, see AXIS7.md#10's open items) and no B x rate load axis needed
(routing structure, not load-dependent, per the same reasoning as take 2).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)
from sweep_sglang_depth_width import (
    filter_overlong,
    run_open_loop,
    send_one,
    stop_server,
    summarize,
    wait_for_server,
)
from specloop_rt.workload import homogeneous

HF_HOME = "/root/spec_decode_env/hf_cache"
HOOK_SHIM_DIR = os.path.join(_REPO_ROOT, "specloop_rt", "sglang_patch")


def launch_server_with_verify_hook(model_path: str, draft_path: str, num_steps: int,
                                    topk: int, num_draft_tokens: int, context_length: int,
                                    max_running_requests: int, port: int, log_path: str,
                                    hook_out_path: str, cell_tag: str, num_layers: int,
                                    topk_size: int, dtype: str = "bfloat16") -> subprocess.Popen:
    """Same launch as sweep_sglang_depth_width.launch_server, plus the env
    vars verify_batch_expert_hooks.py / sitecustomize.py need. See
    specloop_rt/sglang_patch/sitecustomize.py's docstring for why both
    REPO_ROOT and HOOK_SHIM_DIR must be on PYTHONPATH.

    Deliberately does NOT pass --enable-return-routed-experts (take 2's
    flag) -- that flag controls the NATIVE capturer's client-facing return
    path, which this sweep doesn't use at all; it reads the capturer's
    internal device buffer directly via the ModelRunner.forward patch.
    Whether the native capturer even needs to be constructed for this
    patch to see routed_experts_output is confirmed live in the smoke
    test, not assumed here -- see AXIS7.md#10 if this needs revisiting."""
    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME
    env["HUGGINGFACE_HUB_CACHE"] = f"{HF_HOME}/hub"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [_REPO_ROOT, HOOK_SHIM_DIR, existing_pp] if p
    )
    env["CAVEMAN_VERIFY_HOOK_OUT"] = hook_out_path
    env["CAVEMAN_VERIFY_HOOK_CELL_TAG"] = cell_tag
    env["CAVEMAN_VERIFY_HOOK_NUM_LAYERS"] = str(num_layers)
    env["CAVEMAN_VERIFY_HOOK_TOPK_SIZE"] = str(topk_size)
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", draft_path,
        "--speculative-num-steps", str(num_steps),
        "--speculative-eagle-topk", str(topk),
        "--speculative-num-draft-tokens", str(num_draft_tokens),
        "--context-length", str(context_length),
        "--mem-fraction-static", "0.85",
        "--dtype", dtype,
        "--max-running-requests", str(max_running_requests),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--enable-return-routed-experts",  # constructs the capturer at all
    ]
    logf = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)


def aggregate_hook_log(hook_out_path: str) -> dict:
    """Fold a cell's raw per-(verify-step, layer) JSONL into summary
    stats. Returns {} if missing/empty -- the expected outcome for a dense
    control or a broken hook, not silently treated as success either way
    (run_cell()'s WARNING check catches the latter)."""
    if not os.path.exists(hook_out_path):
        return {}
    distinct_experts, max_per_expert, tokens_routed, batch_sizes = [], [], [], []
    with open(hook_out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            distinct_experts.append(row["distinct_experts"])
            max_per_expert.append(row["max_tokens_per_expert"])
            tokens_routed.append(row["tokens_routed"])
            batch_sizes.append(row["verify_batch_tokens"])
    if not distinct_experts:
        return {}
    import numpy as np
    de = np.array(distinct_experts, dtype=float)
    mpe = np.array(max_per_expert, dtype=float)
    tr = np.array(tokens_routed, dtype=float)
    bs = np.array(batch_sizes, dtype=float)
    imbalance = mpe / np.maximum(tr / np.maximum(de, 1), 1e-9)
    return {
        "verify_layer_calls": int(len(distinct_experts)),
        "mean_verify_batch_tokens": float(bs.mean()),
        "mean_distinct_experts": float(de.mean()),
        "p95_distinct_experts": float(np.percentile(de, 95)),
        "mean_max_tokens_per_expert": float(mpe.mean()),
        "mean_imbalance": float(imbalance.mean()),
    }


def run_cell(model_path, draft_path, num_steps, topk, num_draft_tokens,
             context_length, batch, rate, duration, rtype, max_new_tokens,
             port, out_dir, tag, num_layers, topk_size, dtype="bfloat16"):
    effective_draft_tokens = min(num_draft_tokens, num_steps * topk + 1)
    log_path = os.path.join(out_dir, f"server_{tag}.log")
    hook_out_path = os.path.join(out_dir, f"hooklog_{tag}.jsonl")
    if os.path.exists(hook_out_path):
        os.remove(hook_out_path)
    proc = launch_server_with_verify_hook(model_path, draft_path, num_steps, topk,
                                          effective_draft_tokens, context_length, batch,
                                          port, log_path, hook_out_path, tag,
                                          num_layers, topk_size, dtype=dtype)
    try:
        wait_for_server(port, proc)
        trace = homogeneous(rtype=rtype, rate=rate, duration=duration, seed=0,
                            use_real_corpus=True)
        trace = filter_overlong(trace, context_length, max_new_tokens)
        send_one(port, trace[0].prompt, 8)
        rows = run_open_loop(port, trace, max_new_tokens)
        summary = summarize(rows)
    finally:
        stop_server(proc, port)
        # verify_batch_expert_hooks._record() writes to hook_out_path
        # WRITE-THROUGH (one open+append+close per verify step), not
        # buffered-and-atexit -- the original atexit-based design didn't
        # survive stop_server()'s SIGTERM (raw SIGTERM doesn't run atexit
        # handlers, confirmed live: canaries showed real verify-batch data
        # arriving at the hook, but nothing reached disk under the old
        # buffered design -- see verify_batch_expert_hooks.py's _record()
        # docstring and AXIS7.md#10). stop_server() waiting for the
        # process to exit is still correct here, just no longer load-
        # bearing for data completeness the way it was meant to be.
    footprint = aggregate_hook_log(hook_out_path)
    if not footprint:
        print(f"WARNING: {tag} produced no verify-batch expert rows "
              f"(hook not firing? check server_{tag}.log for canary prints "
              f"and import errors)", flush=True)
    return {**summary, **footprint}


def main(argv=None):
    p = argparse.ArgumentParser("SGLang FULL verify-batch expert-footprint sweep "
                                "(Axis-7 C1, take 3)")
    p.add_argument("--model-path", required=True)
    p.add_argument("--draft-path", required=True)
    p.add_argument("--num-layers", type=int, required=True,
                   help="target model's num_hidden_layers -- check config.json, don't "
                        "assume a default is current")
    p.add_argument("--topk-size", type=int, required=True,
                   help="target model's num_experts_per_tok")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--topks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-draft-tokens", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--Bs", type=int, nargs="+", default=[16])
    p.add_argument("--rates", type=float, nargs="+", default=[4.0])
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--rtype", default="code")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--port", type=int, default=30030)
    p.add_argument("--out", default="results_gpu_sweep/sglang_verify_footprint_qwen3moe")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    grid_path = os.path.join(a.out, "grid.json")
    grid = json.load(open(grid_path)) if os.path.exists(grid_path) else {}

    for steps in a.steps:
        for topk in a.topks:
            for B in a.Bs:
                for rate in a.rates:
                    key = f"steps{steps}_topk{topk}_B{B}_rate{rate:g}_{a.rtype}"
                    if key in grid and "error" not in grid[key]:
                        print(f".. skip {key}", flush=True)
                        continue
                    print(f">> {key}", flush=True)
                    try:
                        m = run_cell(a.model_path, a.draft_path, steps, topk,
                                    a.num_draft_tokens, a.context_length, B, rate,
                                    a.duration, a.rtype, a.max_new_tokens, a.port,
                                    a.out, key, a.num_layers, a.topk_size, dtype=a.dtype)
                        grid[key] = {"num_steps": steps, "eagle_topk": topk, "B": B,
                                    "rate": rate, "rtype": a.rtype, **m}
                    except Exception as e:
                        print(f"!! {key} FAILED: {e}", flush=True)
                        grid[key] = {"num_steps": steps, "eagle_topk": topk, "B": B,
                                    "rate": rate, "rtype": a.rtype, "error": str(e)}
                    with open(grid_path, "w") as f:
                        json.dump(grid, f, indent=2)

    print("\n=== mean_distinct_experts (FULL verify batch) by (steps, topk) ===")
    for steps in a.steps:
        for topk in a.topks:
            vals = [v["mean_distinct_experts"] for k, v in grid.items()
                    if v.get("num_steps") == steps and v.get("eagle_topk") == topk
                    and "mean_distinct_experts" in v]
            if vals:
                print(f"steps={steps} topk={topk}: {sum(vals)/len(vals):.2f} "
                      f"(n={len(vals)})")


if __name__ == "__main__":
    main()
