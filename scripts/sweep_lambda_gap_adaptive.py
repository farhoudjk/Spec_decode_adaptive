from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time

import requests

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)

from specloop_rt.workload import homogeneous

TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DRAFT_MODEL = "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex"


def filter_overlong(trace: list, context_length: int, max_new_tokens: int) -> list:
    safety_margin_tokens = 100
    budget_chars = max(0, context_length - max_new_tokens - safety_margin_tokens) * 4
    kept = [r for r in trace if len(r.prompt) <= budget_chars]
    dropped = len(trace) - len(kept)
    if dropped:
        print(f"    .. dropped {dropped}/{len(trace)} overlong prompts", flush=True)
    return kept


def wait_for_server(port: int, proc: subprocess.Popen, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited (code {proc.returncode}) before ready on port {port}")
        try:
            r = requests.get(f"http://localhost:{port}/get_model_info", timeout=5)
            if r.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"server on port {port} not up within {timeout_s}s")


def wait_for_port_free(port: int, timeout_s: int = 60) -> None:
    import socket
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex(("localhost", port)) != 0:
                return
        time.sleep(1)
    print(f"    WARNING: port {port} still in use after {timeout_s}s", flush=True)


def launch_server(num_steps, topk, num_draft_tokens, context_length, batch,
                  port, log_path, cuda_home, hf_home, adaptive_cfg_path=None,
                  model_path=None, draft_path=None, moe_runner_backend=None,
                  mem_fraction_static=0.85):
    env = os.environ.copy()
    env["HF_HOME"] = hf_home
    env["HUGGINGFACE_HUB_CACHE"] = f"{hf_home}/hub"
    env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")
    env["CUDA_HOME"] = cuda_home
    env["SGLANG_CACHE_DIR"] = os.environ.get("SGLANG_CACHE_DIR", "/mnt/data/sglang_cache")

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path or TARGET_MODEL,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", draft_path or DRAFT_MODEL,
        "--speculative-num-steps", str(num_steps),
        "--speculative-eagle-topk", str(topk),
        "--speculative-num-draft-tokens", str(num_draft_tokens),
        "--context-length", str(context_length),
        "--attention-backend", "triton",
        "--mem-fraction-static", str(mem_fraction_static),
        "--max-running-requests", str(batch),
        "--port", str(port),
        "--host", "0.0.0.0",
    ]
    if moe_runner_backend:
        cmd += ["--moe-runner-backend", moe_runner_backend]
    if adaptive_cfg_path is not None:
        cmd += ["--speculative-adaptive",
                "--speculative-adaptive-config", adaptive_cfg_path]
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


def send_one(port, prompt, max_new_tokens, arrival_s=None, request_timeout_s=180):
    submit_wall = time.monotonic()
    r = requests.post(
        f"http://localhost:{port}/generate",
        json={"text": prompt,
              "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens}},
        timeout=request_timeout_s,
    )
    wall = time.monotonic() - submit_wall
    r.raise_for_status()
    mi = r.json()["meta_info"]
    return {
        "wall_s": wall,
        "arrival_s": arrival_s,
        "completion_tokens": mi.get("completion_tokens"),
        "spec_accept_length": mi.get("spec_accept_length"),
        "spec_verify_ct": mi.get("spec_verify_ct"),
    }


def run_open_loop(port, trace, max_new_tokens, request_timeout_s=180):
    t0 = time.monotonic()
    results = [None] * len(trace)

    def submit_one(i, req):
        dt = req.arrival_s - (time.monotonic() - t0)
        if dt > 0:
            time.sleep(dt)
        n_tok = min(req.max_tokens, max_new_tokens)
        try:
            results[i] = send_one(port, req.prompt, n_tok, arrival_s=req.arrival_s,
                                  request_timeout_s=request_timeout_s)
        except Exception as e:
            results[i] = {"error": repr(e)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(trace))) as ex:
        futs = [ex.submit(submit_one, i, req) for i, req in enumerate(trace)]
        for f in futs:
            f.result()
    wall_total = time.monotonic() - t0
    return [r for r in results if r is not None], wall_total


def summarize(rows, wall_total):
    import numpy as np
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    if not ok:
        return {"n": 0, "n_err": len(errs), "per_token_s_p50": None,
                "tok_per_s_per_req": None, "agg_throughput_tok_s": 0.0}
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    wall = np.array([r["wall_s"] for r in ok], dtype=float)
    tok = np.array([r["completion_tokens"] or 0 for r in ok], dtype=float)
    per_token_s = wall / np.maximum(tok, 1)
    ptps50 = float(np.percentile(per_token_s, 50))
    acc = [r["spec_accept_length"] for r in ok if r["spec_accept_length"] is not None]
    vct = [r["spec_verify_ct"] for r in ok if r["spec_verify_ct"] is not None]
    return {
        "n": len(ok),
        "n_err": len(errs),
        "per_token_s_p50": ptps50,
        "tok_per_s_per_req": float(1.0 / ptps50) if ptps50 > 0 else None,
        "agg_throughput_tok_s": float(out_tok / wall_total) if wall_total > 0 else 0.0,
        "out_tok": int(out_tok),
        "wall_total_s": float(wall_total),
        "e2e_p50": float(np.percentile(wall, 50)),
        "e2e_p95": float(np.percentile(wall, 95)),
        "mean_accept_length": float(np.mean(acc)) if acc else None,
        "mean_verify_ct": float(np.mean(vct)) if vct else None,
    }


_STEP_SWITCH_RE = re.compile(r"Adaptive spec params updated: steps (\d+) -> (\d+)")
_INIT_RE = re.compile(r"AdaptiveSpeculativeParams initialized: steps=(\d+)")


def parse_adaptive_trajectory(log_path):
    try:
        with open(log_path) as f:
            text = f.read()
    except OSError:
        return {"initial_steps": None, "switches": [], "final_steps": None, "n_switches": 0}
    init = _INIT_RE.search(text)
    initial = int(init.group(1)) if init else None
    switches = [(int(a), int(b)) for a, b in _STEP_SWITCH_RE.findall(text)]
    final = switches[-1][1] if switches else initial
    unsupported = "adaptive" in text.lower() and "not supported" in text.lower()
    return {
        "initial_steps": initial,
        "switches": switches,
        "n_switches": len(switches),
        "final_steps": final,
        "possibly_unsupported": unsupported,
    }


def run_cell(label, num_steps, topk, batch, rate, args, adaptive_cfg_path=None):
    num_draft_tokens = min(args.num_draft_tokens, num_steps * topk + 1)
    log_path = os.path.join(args.out_dir, f"server_{label}.log")
    print(f"\n  [cell] {label}: D={num_steps} W={topk} B={batch} lambda={rate} "
          f"draft_tokens={num_draft_tokens} adaptive={adaptive_cfg_path is not None}", flush=True)
    proc = launch_server(num_steps, topk, num_draft_tokens, args.context_length,
                         batch, args.port, log_path, args.cuda_home, args.hf_home,
                         adaptive_cfg_path=adaptive_cfg_path,
                         model_path=getattr(args, 'model_path', None),
                         draft_path=getattr(args, 'draft_path', None),
                         moe_runner_backend=getattr(args, 'moe_runner_backend', None),
                         mem_fraction_static=getattr(args, 'mem_fraction_static', 0.85))
    try:
        wait_for_server(args.port, proc, timeout_s=args.boot_timeout)
        trace = homogeneous(rtype="code", rate=rate, duration=args.duration,
                            seed=0, use_real_corpus=True)
        trace = filter_overlong(trace, args.context_length, args.max_new_tokens)
        send_one(args.port, trace[0].prompt, 8)
        rows, wall_total = run_open_loop(args.port, trace, args.max_new_tokens,
                                         request_timeout_s=args.request_timeout)
        summary = summarize(rows, wall_total)
    finally:
        stop_server(proc, args.port)

    tps = summary.get("tok_per_s_per_req")
    tps_s = f"{tps:.1f}" if tps is not None else "n/a"
    agg = summary.get("agg_throughput_tok_s")
    if adaptive_cfg_path is not None:
        summary["adaptive"] = parse_adaptive_trajectory(log_path)
        a = summary["adaptive"]
        print(f"    -> per-req={tps_s} tok/s  agg={agg:.0f} tok/s  "
              f"adaptive: init={a['initial_steps']} final={a['final_steps']} "
              f"switches={a['n_switches']}", flush=True)
    else:
        al = summary.get("mean_accept_length")
        al_s = f"{al:.2f}" if al is not None else "n/a"
        print(f"    -> per-req={tps_s} tok/s  agg={agg:.0f} tok/s  "
              f"accept_len={al_s}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+", default=[6, 8, 10])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--batches", type=int, nargs="+", default=[24, 32])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--moe-runner-backend", default=None)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--draft-path", default=None)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--num-draft-tokens", type=int, default=13)
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--boot-timeout", type=int, default=900)
    ap.add_argument("--request-timeout", type=float, default=180)
    ap.add_argument("--cuda-home", default="/mnt/data/sglang_venv/lib/python3.12/site-packages/nvidia/cu13")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/mnt/data/hf_cache"))
    ap.add_argument("--port", type=int, default=30100)
    ap.add_argument("--out-dir", default="results_gpu_sweep/lambda_gap_adaptive")
    ap.add_argument("--skip-adaptive", action="store_true")
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    adaptive_cfg = {
        str(bs): {
            "candidate_steps": sorted(args.depths),
            "up_hysteresis": 0.0,
            "down_hysteresis": -0.25,
            "ceiling_coeff": 0,
        }
        for bs in (1, 8, 32, 64)
    }
    adaptive_cfg_path = os.path.join(args.out_dir, "adaptive_config.json")
    with open(adaptive_cfg_path, "w") as f:
        json.dump(adaptive_cfg, f, indent=2)
    print(f"[gap] wrote adaptive config (candidate_steps={sorted(args.depths)} "
          f"at every BS slot) to {adaptive_cfg_path}", flush=True)

    out_path = os.path.join(args.out_dir, "result.json")
    results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                results = json.load(f)
            print(f"[gap] resuming: {len(results)} load point(s) already in "
                  f"{out_path}: {sorted(results)}", flush=True)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[gap] WARNING: could not read {out_path} ({e}); "
                  f"starting fresh", flush=True)

    for batch in args.batches:
        for rate in args.lambdas:
            key = f"B{batch}_lam{rate:g}"
            results.setdefault(key, {})

            if not args.skip_static:
                for d in args.depths:
                    label = f"{key}_static_D{d}W{args.topk}"
                    results[key][f"static_D{d}"] = run_cell(
                        label, d, args.topk, batch, rate, args)
                    with open(out_path, "w") as f:
                        json.dump(results, f, indent=2)

            if not args.skip_adaptive:
                label = f"{key}_adaptive"
                results[key]["adaptive"] = run_cell(
                    label, max(args.depths), args.topk, batch, rate, args,
                    adaptive_cfg_path=adaptive_cfg_path)
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\n[gap] wrote {out_path}", flush=True)

    def _m(cell_entry):
        return (cell_entry or {}).get("tok_per_s_per_req")

    print(f"\n{'='*84}\n[gap] PER-REQUEST DECODE SPEED BY (B,lambda) x D  [W={args.topk}]\n{'='*84}", flush=True)
    hdr = f"{'load point':14s}" + "".join(f"{'D'+str(d):>9s}" for d in args.depths) \
          + f"{'BEST':>11s}{'adaptive':>10s}{'adapt D':>9s}"
    print(hdr, flush=True)
    for key, cell in results.items():
        row = f"{key:14s}"
        statics = []
        for d in args.depths:
            g = _m(cell.get(f"static_D{d}"))
            statics.append((g, d))
            row += f"{g:>9.1f}" if g is not None else f"{'-':>9s}"
        valid = [(g, d) for g, d in statics if g is not None]
        if valid:
            bg, bd = max(valid)
            row += f"{('D'+str(bd)+':'+f'{bg:.0f}'):>11s}"
        else:
            row += f"{'-':>11s}"
        ad = cell.get("adaptive", {})
        ag = _m(ad)
        row += f"{ag:>10.1f}" if ag is not None else f"{'-':>10s}"
        af = ad.get("adaptive", {}).get("final_steps")
        row += f"{str(af):>9s}"
        print(row, flush=True)

    print(f"\n{'='*84}\n[gap] DOES THE BUILT-IN CONTROLLER FIND THE OPTIMUM?\n{'='*84}", flush=True)
    for key, cell in results.items():
        statics = [(_m(cell.get(f"static_D{d}")), d) for d in args.depths]
        valid = [(g, d) for g, d in statics if g is not None]
        ad = cell.get("adaptive", {})
        ag, traj = _m(ad), ad.get("adaptive", {})
        if not valid or ag is None:
            continue
        bg, bd = max(valid)
        gap = 100 * (1 - ag / bg) if bg else float("nan")
        print(f"  {key:14s} best=D{bd} ({bg:.1f})  adaptive={ag:.1f} "
              f"(settled D={traj.get('final_steps')}, {traj.get('n_switches')} switches) "
              f"-> leaves {gap:+.1f}% on the table", flush=True)

    print(f"\n{'='*84}\n[gap] CROSS-CHECK: does aggregate throughput agree on best D?\n{'='*84}", flush=True)
    for key, cell in results.items():
        per_req = [(_m(cell.get(f"static_D{d}")), d) for d in args.depths]
        agg = [((cell.get(f"static_D{d}") or {}).get("agg_throughput_tok_s"), d)
               for d in args.depths]
        pv = [(g, d) for g, d in per_req if g is not None]
        av = [(g, d) for g, d in agg if g is not None]
        if not pv or not av:
            continue
        _, bd_p = max(pv)
        _, bd_a = max(av)
        verdict = "AGREE" if bd_p == bd_a else "DISAGREE"
        print(f"  {key:14s} per-request best=D{bd_p}   aggregate best=D{bd_a}   -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
