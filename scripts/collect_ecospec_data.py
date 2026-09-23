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


def launch_server(model_path, draft_path, num_steps, topk, context_length,
                   max_running_requests, port, log_path, draft_out_path,
                   verify_out_path, topk_size, dtype="bfloat16",
                   attention_backend=None, moe_runner_backend=None,
                   sampling_backend=None, mem_fraction_static=0.85,
                   cuda_home=None) -> subprocess.Popen:
    total_width = topk + topk * topk * (num_steps - 1)
    num_draft_tokens = total_width + 1

    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME
    env["HUGGINGFACE_HUB_CACHE"] = f"{HF_HOME}/hub"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [_REPO_ROOT, HOOK_SHIM_DIR, existing_pp] if p
    )
    if cuda_home:
        env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")
        env["CUDA_HOME"] = cuda_home
    env["CAVEMAN_ECOSPEC_DRAFT_OUT"] = draft_out_path
    env["CAVEMAN_ECOSPEC_VERIFY_OUT"] = verify_out_path
    env["CAVEMAN_ECOSPEC_TOPK_SIZE"] = str(topk_size)
    env["CAVEMAN_ECOSPEC_EAGLE_TOPK"] = str(topk)

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
    print(f"  [ecospec-collect] D={num_steps} W={topk} num_draft_tokens={num_draft_tokens}",
          flush=True)
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--draft-path", required=True)
    p.add_argument("--topk-size", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--eagle-topk", type=int, required=True)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--rate", type=float, default=2.0)
    p.add_argument("--n-requests", type=int, default=60)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--rtype", default="code")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attention-backend", default=None)
    p.add_argument("--moe-runner-backend", default=None)
    p.add_argument("--sampling-backend", default=None)
    p.add_argument("--mem-fraction-static", type=float, default=0.85)
    p.add_argument("--cuda-home", default=None)
    p.add_argument("--request-timeout", type=float, default=180)
    p.add_argument("--port", type=int, default=30960)
    p.add_argument("--out", default="results_gpu_sweep/ecospec_collect")
    a = p.parse_args(argv)
    if a.eagle_topk < 2:
        p.error("--eagle-topk must be >=2")

    os.makedirs(a.out, exist_ok=True)
    tag = f"D{a.steps}W{a.eagle_topk}"
    draft_out = os.path.join(a.out, f"draft_{tag}.jsonl")
    verify_out = os.path.join(a.out, f"verify_{tag}.jsonl")
    log_path = os.path.join(a.out, f"server_{tag}.log")
    open(draft_out, "w").close()
    open(verify_out, "w").close()

    proc = launch_server(a.model_path, a.draft_path, a.steps, a.eagle_topk,
                         a.context_length, a.B, a.port, log_path, draft_out,
                         verify_out, a.topk_size, dtype=a.dtype,
                         attention_backend=a.attention_backend,
                         moe_runner_backend=a.moe_runner_backend,
                         sampling_backend=a.sampling_backend,
                         mem_fraction_static=a.mem_fraction_static,
                         cuda_home=a.cuda_home)
    per_request_out = os.path.join(a.out, f"perrequest_{tag}.jsonl")
    open(per_request_out, "w").close()
    try:
        wait_for_server(a.port, proc, timeout_s=900)
        duration = a.n_requests / a.rate
        trace = homogeneous(rtype=a.rtype, rate=a.rate, duration=duration,
                            seed=0, use_real_corpus=True)
        trace = filter_overlong(trace, a.context_length, a.max_new_tokens)
        send_one(a.port, trace[0].prompt, 8, request_timeout_s=a.request_timeout)

        rows = run_open_loop(a.port, trace, a.max_new_tokens,
                             request_timeout_s=a.request_timeout)

        verify_all = []
        if os.path.exists(verify_out):
            with open(verify_out) as vf:
                for line in vf:
                    line = line.strip()
                    if line:
                        verify_all.append(json.loads(line))

        with open(per_request_out, "a") as out_f:
            for i, r in enumerate(rows):
                out_f.write(json.dumps({
                    "req_id": i,
                    "wall_s": r.get("wall_s"),
                    "completion_tokens": r.get("completion_tokens"),
                    "spec_accept_rate": r.get("spec_accept_rate"),
                    "spec_accept_length": r.get("spec_accept_length"),
                    "spec_verify_ct": r.get("spec_verify_ct"),
                }) + "\n")

        summary = summarize(rows)
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
        print(f"[ecospec-collect] wrote {len(rows)} per-request rows to {per_request_out}, "
              f"{len(verify_all)} verify-steps to {verify_out}")
    finally:
        stop_server(proc, a.port)


if __name__ == "__main__":
    main()
