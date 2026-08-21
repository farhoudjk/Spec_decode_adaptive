"""Real-GPU goodput comparison: stock SGLang EAGLE3 tree construction vs.
footprint-aware draft pruning (idea 1), on real load.

Everything up to this script validated CORRECTNESS and STABILITY of the
live hook (specloop_rt/sglang_patch/footprint_aware_pruning.py +
install_footprint_pruning.py): lambda=0 identity confirmed byte-for-byte
against an unpatched baseline, lambda=1/50 ran with zero exceptions and
coherent output. This script is the first to measure whether the
mechanism actually changes THROUGHPUT under real concurrent load --
nothing before this point tested that.

Method, matching this repo's own sweep convention (scripts/
sweep_sglang_depth_width.py, AXIS6/7's own evaluation approach): launch a
real SGLang server, drive it with an open-loop Poisson arrival trace of
real prompts (specloop_rt.workload.homogeneous, real_corpus-backed), and
compare summarized per-token latency / acceptance stats between two runs
that differ ONLY in whether CAVEMAN_PRUNING_HOOK_OUT is set (same tree
shape, same model, same trace, same seed).

Runs on sglang==0.4.10 (this repo's A100-compatible environment -- see
install_footprint_pruning.py's docstring for why 0.5.17 doesn't work on
this GPU). Endpoint names differ from the newer sweep script's
assumptions (/get_model_info, not /model_info -- confirmed live this
session) -- this script uses the correct ones for 0.4.10, not copied
blindly from the other sweep script.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time

import requests

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)

from specloop_rt.workload import homogeneous

HF_HOME_DEFAULT = "/root/hf_cache"
TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DRAFT_MODEL = "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex"

REPO_ROOT_ENV = _REPO_ROOT
SGLANG_PATCH_DIR = os.path.join(_REPO_ROOT, "specloop_rt", "sglang_patch")


def filter_overlong(trace: list, context_length: int, max_new_tokens: int) -> list:
    """Same conservative estimate as sweep_sglang_depth_width.py's own
    filter_overlong -- kept identical rather than re-derived, since this
    is about trace validity, not something this session's work changes."""
    safety_margin_tokens = 100
    budget_chars = max(0, context_length - max_new_tokens - safety_margin_tokens) * 4
    kept = [r for r in trace if len(r.prompt) <= budget_chars]
    dropped = len(trace) - len(kept)
    if dropped:
        print(f".. dropped {dropped}/{len(trace)} overlong prompts", flush=True)
    return kept


def wait_for_server(port: int, proc: subprocess.Popen, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server process exited (code {proc.returncode}) "
                               f"before becoming ready on port {port}")
        try:
            # /get_model_info, NOT /model_info -- confirmed live this
            # session against sglang 0.4.10 (the 0.5.17-era sweep script's
            # /model_info returns 404 here; a stale watcher hit exactly
            # this earlier in the session before being caught).
            r = requests.get(f"http://localhost:{port}/get_model_info", timeout=5)
            if r.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"server on port {port} did not come up within {timeout_s}s")


def launch_server(num_steps: int, topk: int, num_draft_tokens: int, context_length: int,
                  max_running_requests: int, port: int, log_path: str,
                  pruning_enabled: bool, ckpt_path: str, lam: float, max_batch: int,
                  hf_home: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["HF_HOME"] = hf_home
    env["HUGGINGFACE_HUB_CACHE"] = f"{hf_home}/hub"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_XET_HIGH_PERFORMANCE"] = "0"
    if pruning_enabled:
        env["PYTHONPATH"] = f"{REPO_ROOT_ENV}:{SGLANG_PATCH_DIR}"
        env["CAVEMAN_PRUNING_HOOK_OUT"] = "1"
        env["CAVEMAN_PRUNING_CKPT_PATH"] = ckpt_path
        env["CAVEMAN_PRUNING_LAMBDA"] = str(lam)
        env["CAVEMAN_PRUNING_MAX_BATCH"] = str(max_batch)
    else:
        # Explicitly ensure a clean baseline run doesn't inherit pruning
        # env vars from the calling shell (defense against a leftover
        # export from earlier manual testing this session).
        for k in ["CAVEMAN_PRUNING_HOOK_OUT", "CAVEMAN_PRUNING_CKPT_PATH",
                  "CAVEMAN_PRUNING_LAMBDA", "CAVEMAN_PRUNING_MAX_BATCH", "PYTHONPATH"]:
            env.pop(k, None)

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", TARGET_MODEL,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", DRAFT_MODEL,
        "--speculative-num-steps", str(num_steps),
        "--speculative-eagle-topk", str(topk),
        "--speculative-num-draft-tokens", str(num_draft_tokens),
        "--context-length", str(context_length),
        "--mem-fraction-static", "0.85",
        "--dtype", "bfloat16",
        "--max-running-requests", str(max_running_requests),
        "--port", str(port),
        "--host", "0.0.0.0",
    ]
    logf = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)


def stop_server(proc: subprocess.Popen, port: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    wait_for_port_free(port)


def wait_for_port_free(port: int, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex(("localhost", port)) != 0:
                return
        time.sleep(1)
    print(f"WARNING: port {port} still in use after {timeout_s}s", flush=True)


def send_one(port: int, prompt: str, max_new_tokens: int, arrival_s: float = None,
            request_timeout_s: float = 180) -> dict:
    submit_wall = time.monotonic()
    r = requests.post(
        f"http://localhost:{port}/generate",
        json={"text": prompt,
              "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens}},
        timeout=request_timeout_s,
    )
    wall = time.monotonic() - submit_wall
    r.raise_for_status()
    d = r.json()
    mi = d["meta_info"]
    return {
        "wall_s": wall,
        "arrival_s": arrival_s,
        "e2e_latency": mi.get("e2e_latency"),
        "completion_tokens": mi.get("completion_tokens"),
        "spec_accept_rate": mi.get("spec_accept_rate"),
        "spec_accept_length": mi.get("spec_accept_length"),
        "spec_verify_ct": mi.get("spec_verify_ct"),
    }


def run_open_loop(port: int, trace: list, max_new_tokens: int,
                  request_timeout_s: float = 180) -> list:
    t0 = time.monotonic()
    results = [None] * len(trace)

    def submit_one(i, req):
        dt = req.arrival_s - (time.monotonic() - t0)
        if dt > 0:
            time.sleep(dt)
        n_tok = min(req.max_tokens, max_new_tokens)
        results[i] = send_one(port, req.prompt, n_tok, arrival_s=req.arrival_s,
                               request_timeout_s=request_timeout_s)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(trace))) as ex:
        futs = [ex.submit(submit_one, i, req) for i, req in enumerate(trace)]
        for f in futs:
            f.result()
    return [r for r in results if r is not None]


def summarize(rows: list) -> dict:
    import numpy as np
    if not rows:
        return {"n": 0}
    wall = np.array([r["wall_s"] for r in rows])
    tok = np.array([r["completion_tokens"] for r in rows if r["completion_tokens"]])
    accept_rate = [r["spec_accept_rate"] for r in rows if r["spec_accept_rate"] is not None]
    accept_len = [r["spec_accept_length"] for r in rows if r["spec_accept_length"] is not None]
    verify_ct = [r["spec_verify_ct"] for r in rows if r["spec_verify_ct"] is not None]
    per_token_s = wall / np.maximum(tok, 1) if len(tok) == len(wall) else wall
    total_tokens = float(tok.sum()) if len(tok) else 0.0
    total_wall = float(wall.sum())

    # SERVER-truth latency (meta_info's own e2e_latency, measured inside
    # sglang around just the generate() call) alongside client wall_s
    # (includes HTTP round-trip + this harness's own ThreadPoolExecutor
    # dispatch jitter) -- kept SEPARATE so a client-harness measurement
    # artifact can be told apart from a genuine server-side slowdown,
    # rather than conflated into one goodput number the way the original
    # version of this script did.
    e2e = np.array([r["e2e_latency"] for r in rows if r.get("e2e_latency") is not None])
    e2e_tok = np.array([r["completion_tokens"] for r in rows
                        if r.get("e2e_latency") is not None and r.get("completion_tokens")])
    total_e2e = float(e2e.sum()) if len(e2e) else 0.0
    total_e2e_tok = float(e2e_tok.sum()) if len(e2e_tok) else 0.0

    return {
        "n": len(rows),
        "wall_p50": float(np.percentile(wall, 50)),
        "wall_p95": float(np.percentile(wall, 95)),
        "per_token_s_p50": float(np.percentile(per_token_s, 50)),
        "goodput_tok_per_s": total_tokens / total_wall if total_wall > 0 else None,
        "goodput_tok_per_s_server_e2e": total_e2e_tok / total_e2e if total_e2e > 0 else None,
        "e2e_p50": float(np.percentile(e2e, 50)) if len(e2e) else None,
        "e2e_p95": float(np.percentile(e2e, 95)) if len(e2e) else None,
        "mean_accept_rate": float(np.mean(accept_rate)) if accept_rate else None,
        "mean_accept_length": float(np.mean(accept_len)) if accept_len else None,
        "mean_verify_ct": float(np.mean(verify_ct)) if verify_ct else None,
    }


def run_cell(num_steps, topk, num_draft_tokens, context_length, batch, rate, duration,
            rtype, max_new_tokens, port, out_dir, tag, pruning_enabled, ckpt_path, lam,
            max_batch, hf_home, request_timeout_s, seed):
    log_path = os.path.join(out_dir, f"server_{tag}.log")
    proc = launch_server(num_steps, topk, num_draft_tokens, context_length, batch, port,
                         log_path, pruning_enabled, ckpt_path, lam, max_batch, hf_home)
    try:
        wait_for_server(port, proc)
        trace = homogeneous(rtype=rtype, rate=rate, duration=duration, seed=seed,
                            use_real_corpus=True)
        trace = filter_overlong(trace, context_length, max_new_tokens)
        send_one(port, trace[0].prompt, 8, request_timeout_s=request_timeout_s)
        rows = run_open_loop(port, trace, max_new_tokens, request_timeout_s=request_timeout_s)
        # Save raw per-request rows, not just the summary -- needed to
        # tell apart a genuine server-side slowdown from a client-harness
        # measurement artifact (wall_s includes HTTP + ThreadPoolExecutor
        # dispatch jitter; e2e_latency is the server's own precise
        # measurement) after the fact, which the first version of this
        # script's summary-only output made impossible to check.
        rows_path = os.path.join(out_dir, f"rows_{tag}.json")
        with open(rows_path, "w") as f:
            json.dump(rows, f)
        return summarize(rows)
    finally:
        stop_server(proc, port)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--num-draft-tokens", type=int, default=8)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--Bs", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--rates", type=float, nargs="+", default=[2.0, 6.0])
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--rtype", default="code")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 20.0])
    ap.add_argument("--ckpt-path", default="results_gpu_sweep/routing_proxy_train/routing_proxy_head.pt")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--out", default="results_gpu_sweep/goodput_pruning_comparison")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", HF_HOME_DEFAULT))
    ap.add_argument("--request-timeout", type=float, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="If given, overrides --seed: run each (B,rate,config) cell once per "
                         "seed in this list and report mean/std across repeats -- needed to "
                         "tell a real small effect (e.g. a few percent goodput) apart from "
                         "single-run noise at this trace duration/sample size.")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    grid_path = os.path.join(a.out, "grid.json")
    grid = json.load(open(grid_path)) if os.path.exists(grid_path) else {}

    ckpt_abs = os.path.abspath(os.path.join(_REPO_ROOT, a.ckpt_path))
    if not os.path.exists(ckpt_abs):
        print(f"FATAL: checkpoint not found at {ckpt_abs}", flush=True)
        sys.exit(1)

    configs = [("baseline", False, 0.0)] + [(f"lam{lam:g}", True, lam) for lam in a.lambdas]
    seeds = a.seeds if a.seeds is not None else [a.seed]

    for B in a.Bs:
        for rate in a.rates:
            for cfg_name, pruning_enabled, lam in configs:
                per_seed_results = []
                for seed in seeds:
                    key = f"B{B}_rate{rate:g}_{cfg_name}_seed{seed}"
                    if key in grid and "error" not in grid[key]:
                        print(f".. skip {key}", flush=True)
                        per_seed_results.append(grid[key])
                        continue
                    print(f">> {key}", flush=True)
                    try:
                        m = run_cell(a.steps, a.topk, a.num_draft_tokens, a.context_length,
                                    B, rate, a.duration, a.rtype, a.max_new_tokens, a.port,
                                    a.out, key, pruning_enabled, ckpt_abs, lam, a.max_batch,
                                    a.hf_home, a.request_timeout, seed)
                        grid[key] = {"B": B, "rate": rate, "config": cfg_name, "seed": seed,
                                    "pruning_enabled": pruning_enabled, "lambda": lam, **m}
                        per_seed_results.append(grid[key])
                    except Exception as e:
                        print(f"!! {key} FAILED: {e}", flush=True)
                        grid[key] = {"B": B, "rate": rate, "config": cfg_name, "seed": seed,
                                    "pruning_enabled": pruning_enabled, "lambda": lam, "error": str(e)}
                    with open(grid_path, "w") as f:
                        json.dump(grid, f, indent=2)

                # Aggregate across seeds into a summary key (mean/std),
                # matching the original single-seed grid's key naming so
                # the print-out below still works unmodified.
                valid = [r for r in per_seed_results if "error" not in r and r.get("goodput_tok_per_s") is not None]
                if valid:
                    import numpy as np
                    gp = np.array([r["goodput_tok_per_s"] for r in valid])
                    vc = np.array([r["mean_verify_ct"] for r in valid if r.get("mean_verify_ct") is not None])
                    agg_key = f"B{B}_rate{rate:g}_{cfg_name}"
                    grid[agg_key] = {
                        "B": B, "rate": rate, "config": cfg_name,
                        "pruning_enabled": pruning_enabled, "lambda": lam,
                        "n_seeds": len(valid),
                        "goodput_tok_per_s": float(gp.mean()),
                        "goodput_tok_per_s_std": float(gp.std()),
                        "mean_verify_ct": float(vc.mean()) if len(vc) else None,
                    }
                    with open(grid_path, "w") as f:
                        json.dump(grid, f, indent=2)

    print("\n=== goodput_tok_per_s by (B, rate, config), mean across seeds ===")
    for B in a.Bs:
        for rate in a.rates:
            for cfg_name, _, _ in configs:
                key = f"B{B}_rate{rate:g}_{cfg_name}"
                v = grid.get(key, {})
                if "error" not in v and v.get("goodput_tok_per_s") is not None:
                    std = v.get("goodput_tok_per_s_std")
                    std_str = f" +/-{std:.2f}" if std is not None else ""
                    print(f"B={B} rate={rate:g} {cfg_name:>10s}: "
                          f"{v['goodput_tok_per_s']:.2f}{std_str} tok/s (n_seeds={v.get('n_seeds', 1)})  "
                          f"verify_ct={v.get('mean_verify_ct')}")


if __name__ == "__main__":
    main()
