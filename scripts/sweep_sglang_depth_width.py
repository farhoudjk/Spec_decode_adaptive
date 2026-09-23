from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specloop_rt.workload import homogeneous

HF_HOME = os.environ.get("HF_HOME") or "/root/spec_decode_env/hf_cache"


def filter_overlong(trace: list, context_length: int, max_new_tokens: int) -> list:
    """Drop trace requests whose prompt is too long to fit the server's
    fixed context budget alongside its completion. Needed for workloads
    like `reason` (CNN-DailyMail articles, specloop_rt.real_corpus) where
    raw article length has no cap and a real fraction of prompts (~7% in
    one check) exceed context_length once tokenized -- those requests get
    a 400 from /generate and crash the whole cell's run_open_loop, not
    just that one request.

    Uses a conservative 4-chars-per-token estimate (no tokenizer available
    client-side) and reserves max_new_tokens + a fixed safety margin for
    chat-template/special-token overhead observed in practice (a request
    logged as failing needed 2088 total against a 2048 cap while its raw
    prompt char count implied fewer tokens than that by the 4-char
    estimate alone -- the margin absorbs that gap). This is deliberately
    conservative: some requests just under the real limit may still be
    dropped, which is fine here since the trace only needs to be
    representative, not exhaustive of the corpus.
    """
    safety_margin_tokens = 100
    budget_chars = max(0, context_length - max_new_tokens - safety_margin_tokens) * 4
    kept = [r for r in trace if len(r.prompt) <= budget_chars]
    dropped = len(trace) - len(kept)
    if dropped:
        print(f".. dropped {dropped}/{len(trace)} overlong prompts "
              f"(> ~{budget_chars} chars) to fit context_length={context_length}",
              flush=True)
    return kept


def wait_for_server(port: int, proc: subprocess.Popen, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server process exited (code {proc.returncode}) "
                               f"before becoming ready on port {port}")
        try:
            r = requests.get(f"http://localhost:{port}/model_info", timeout=5)
            if r.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"server on port {port} did not come up within {timeout_s}s")


def launch_server(model_path: str, draft_path: str, num_steps: int, topk: int,
                   num_draft_tokens: int, context_length: int, max_running_requests: int,
                   port: int, log_path: str, dtype: str = "bfloat16",
                   attention_backend: str = None) -> subprocess.Popen:
    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME
    env["HUGGINGFACE_HUB_CACHE"] = f"{HF_HOME}/hub"
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
    ]
    if attention_backend:
        cmd += ["--attention-backend", attention_backend]
    logf = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)


def stop_server(proc: subprocess.Popen, port: int) -> None:
    import signal
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
    import socket
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
        "spec_num_proposed_drafts": mi.get("spec_num_proposed_drafts"),
        "spec_num_correct_drafts": mi.get("spec_num_correct_drafts"),
    }


def run_open_loop(port: int, trace: list, max_new_tokens: int,
                   request_timeout_s: float = 180) -> list:
    """Dispatch requests paced to trace[i].arrival_s (open-loop, matching
    specloop_rt.replay's arrival-time pacing) rather than firing all at once.
    The pool is sized to the trace length so the client is never the
    bottleneck -- the server's own --max-running-requests is the only
    admission cap that should shape the result; throttling client-side
    concurrency too would impose a second, redundant cap and distort pacing
    for any burst deeper than that cap.
    """
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
    per_token_s = wall / np.maximum(tok, 1) if len(tok) == len(wall) else wall
    return {
        "n": len(rows),
        "wall_p50": float(np.percentile(wall, 50)),
        "wall_p95": float(np.percentile(wall, 95)),
        "per_token_s_p50": float(np.percentile(per_token_s, 50)),
        "mean_accept_rate": float(np.mean(accept_rate)) if accept_rate else None,
        "mean_accept_length": float(np.mean(accept_len)) if accept_len else None,
    }


def run_cell(model_path, draft_path, num_steps, topk, num_draft_tokens,
             context_length, batch, rate, duration, rtype, max_new_tokens,
             port, out_dir, tag, dtype="bfloat16", request_timeout_s=180,
             attention_backend=None):
    effective_draft_tokens = min(num_draft_tokens, num_steps * topk + 1)
    log_path = os.path.join(out_dir, f"server_{tag}.log")
    proc = launch_server(model_path, draft_path, num_steps, topk, effective_draft_tokens,
                         context_length, batch, port, log_path, dtype=dtype,
                         attention_backend=attention_backend)
    try:
        wait_for_server(port, proc)
        trace = homogeneous(rtype=rtype, rate=rate, duration=duration, seed=0,
                            use_real_corpus=True)
        trace = filter_overlong(trace, context_length, max_new_tokens)
        send_one(port, trace[0].prompt, 8, request_timeout_s=request_timeout_s)
        rows = run_open_loop(port, trace, max_new_tokens, request_timeout_s=request_timeout_s)
        return summarize(rows)
    finally:
        stop_server(proc, port)


def main(argv=None):
    p = argparse.ArgumentParser("SGLang depth x width x batch cap x rate sweep")
    p.add_argument("--model-path", required=True)
    p.add_argument("--draft-path", required=True)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--topks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-draft-tokens", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attention-backend", default=None,
                   help="Passed through to sglang.launch_server when set. Leave "
                        "unset to reproduce the published grids exactly. Use "
                        "'triton' on boxes where the default FlashInfer backend "
                        "fails to JIT-compile (CCCL/nvcc header mismatch).")
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--Bs", type=int, nargs="+", default=[8, 32])
    p.add_argument("--rates", type=float, nargs="+", default=[8.0],
                   help="req/s, open-loop Poisson arrival (specloop_rt.workload.homogeneous)")
    p.add_argument("--duration", type=float, default=60.0,
                   help="seconds of trace per cell")
    p.add_argument("--rtype", default="code")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--request-timeout", type=float, default=180,
                   help="per-request HTTP client timeout in seconds; raise for "
                        "low-accept-rate drafts where queues drain slowly")
    p.add_argument("--port", type=int, default=30010)
    p.add_argument("--out", default="results_gpu_sweep/sglang_depth_width")
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
                                    a.out, key, dtype=a.dtype,
                                    request_timeout_s=a.request_timeout,
                                    attention_backend=a.attention_backend)
                        grid[key] = {"num_steps": steps, "eagle_topk": topk, "B": B,
                                    "rate": rate, "rtype": a.rtype, **m}
                    except Exception as e:
                        print(f"!! {key} FAILED: {e}", flush=True)
                        grid[key] = {"num_steps": steps, "eagle_topk": topk, "B": B,
                                    "rate": rate, "rtype": a.rtype, "error": str(e)}
                    with open(grid_path, "w") as f:
                        json.dump(grid, f, indent=2)

    print("\n=== per_token_s_p50 by (steps, topk), averaged over B x rate ===")
    for steps in a.steps:
        for topk in a.topks:
            vals = [v["per_token_s_p50"] for k, v in grid.items()
                    if v.get("num_steps") == steps and v.get("eagle_topk") == topk
                    and "error" not in v]
            if vals:
                print(f"steps={steps} topk={topk}: {sum(vals)/len(vals):.5f}s "
                      f"(n={len(vals)})")


if __name__ == "__main__":
    main()
