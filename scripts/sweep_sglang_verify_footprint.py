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

HF_HOME = os.environ.get("HF_HOME") or "/root/spec_decode_env/hf_cache"
HOOK_SHIM_DIR = os.path.join(_REPO_ROOT, "specloop_rt", "sglang_patch")


def launch_server_with_verify_hook(model_path: str, draft_path: str, num_steps: int,
                                    topk: int, num_draft_tokens: int, context_length: int,
                                    max_running_requests: int, port: int, log_path: str,
                                    hook_out_path: str, cell_tag: str, num_layers: int,
                                    topk_size: int, dtype: str = "bfloat16",
                                    moe_runner_backend: str = None,
                                    attention_backend: str = None,
                                    sampling_backend: str = None,
                                    mem_fraction_static: float = 0.85) -> subprocess.Popen:
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
        "--mem-fraction-static", str(mem_fraction_static),
        "--dtype", dtype,
        "--max-running-requests", str(max_running_requests),
        "--port", str(port),
        "--host", "0.0.0.0",
        "--enable-return-routed-experts",
    ]
    if attention_backend:
        cmd += ["--attention-backend", attention_backend]
    if sampling_backend:
        cmd += ["--sampling-backend", sampling_backend]
    if moe_runner_backend:
        cmd += ["--moe-runner-backend", moe_runner_backend]
    logf = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)


def _wait_for_gpu_drain(timeout_s: float = 120.0, idle_mib: int = 2000) -> None:
    import subprocess as _sp
    import time as _t
    deadline = _t.time() + timeout_s
    while _t.time() < deadline:
        try:
            out = _sp.run(["nvidia-smi", "--query-gpu=memory.used",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=10).stdout.strip()
            if out and int(out.splitlines()[0]) < idle_mib:
                return
        except (OSError, ValueError, _sp.SubprocessError):
            return
        _t.sleep(3)
    print(f"    WARNING: GPU still busy after {timeout_s}s; continuing anyway",
          flush=True)


def aggregate_hook_log(hook_out_path: str) -> dict:
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
             port, out_dir, tag, num_layers, topk_size, dtype="bfloat16",
             request_timeout_s=180, moe_runner_backend=None,
             attention_backend=None, sampling_backend=None,
             mem_fraction_static=0.85):
    effective_draft_tokens = min(num_draft_tokens, num_steps * topk + 1)
    log_path = os.path.join(out_dir, f"server_{tag}.log")
    hook_out_path = os.path.join(out_dir, f"hooklog_{tag}.jsonl")
    if os.path.exists(hook_out_path):
        os.remove(hook_out_path)
    proc = launch_server_with_verify_hook(model_path, draft_path, num_steps, topk,
                                          effective_draft_tokens, context_length, batch,
                                          port, log_path, hook_out_path, tag,
                                          num_layers, topk_size, dtype=dtype,
                                          moe_runner_backend=moe_runner_backend,
                                          attention_backend=attention_backend,
                                          sampling_backend=sampling_backend,
                                          mem_fraction_static=mem_fraction_static)
    try:
        wait_for_server(port, proc)
        trace = homogeneous(rtype=rtype, rate=rate, duration=duration, seed=0,
                            use_real_corpus=True)
        trace = filter_overlong(trace, context_length, max_new_tokens)
        send_one(port, trace[0].prompt, 8, request_timeout_s=request_timeout_s)
        rows = run_open_loop(port, trace, max_new_tokens, request_timeout_s=request_timeout_s)
        summary = summarize(rows)
    finally:
        stop_server(proc, port)
        _wait_for_gpu_drain()
    footprint = aggregate_hook_log(hook_out_path)
    if not footprint:
        print(f"WARNING: {tag} produced no verify-batch expert rows", flush=True)
    return {**summary, **footprint}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--draft-path", required=True)
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--topk-size", type=int, required=True)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--topks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-draft-tokens", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attention-backend", default=None)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--Bs", type=int, nargs="+", default=[16])
    p.add_argument("--rates", type=float, nargs="+", default=[4.0])
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--rtype", default="code")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--request-timeout", type=float, default=180)
    p.add_argument("--moe-runner-backend", default=None)
    p.add_argument("--sampling-backend", default=None)
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
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
                                    a.out, key, a.num_layers, a.topk_size, dtype=a.dtype,
                                    request_timeout_s=a.request_timeout,
                                    moe_runner_backend=a.moe_runner_backend,
                                    attention_backend=a.attention_backend,
                                    sampling_backend=a.sampling_backend,
                                    mem_fraction_static=a.mem_fraction_static)
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
