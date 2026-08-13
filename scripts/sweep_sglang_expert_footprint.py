"""Axis-7 / C1: expert-activation footprint sweep for SGLang EAGLE3 on
Qwen3-MoE. See AXIS7.md for why this exists and AXIS7.md#7 for the pivot
this version reflects.

REWRITTEN AWAY FROM A CUSTOM MONKEYPATCH. The first version of this sweep
patched TopK.forward (specloop_rt/sglang_patch/) to record topk_ids.
GPU-tested and it turned out to install correctly (confirmed via canary
prints in every spawned process) but never fire: SGLang captures a CUDA
graph for the verify forward pass (confirmed via the server's own
`cuda_graph={target_verify=...}` startup timing) and replays it on every
real request thereafter, which bypasses Python-level monkeypatches
entirely -- a graph replay executes the recorded CUDA kernel sequence
directly, never re-entering the Python call that a `TopK.forward`
monkeypatch intercepts.

SGLang already ships a graph-safe, first-class mechanism for exactly this
signal: `--enable-return-routed-experts` (server_args.py) plus a per-request
`return_routed_experts: true` field on /generate. Internally
(state_capturer/routed_experts.py) it writes topk_ids into a pre-allocated
device buffer via a plain in-place tensor write
(`self.buffer[:batch, layer_id, :] = topk_indices`, state_capturer/base.py)
-- exactly the write pattern CUDA graph capture supports natively, so it
survives graph replay where a Python monkeypatch cannot. The result is
base64-encoded int32 and returned in meta_info["routed_experts"]
(managers/detokenizer_manager.py), decoded here with
np.frombuffer + reshape to (num_tokens, num_layers, topk_size).

specloop_rt/sglang_patch/ (the monkeypatch + sitecustomize.py) is dead code
after this rewrite -- left in place for the record (AXIS7.md documents the
dead end and why), not deleted, since the debugging trail (canary prints,
CUDA graph timing correlation) is itself useful provenance for anyone
hitting the same monkeypatch-vs-CUDA-graph trap on a different SGLang hook.

Still a SEPARATE sweep from sweep_sglang_depth_width.py's axis6 grid (not
an in-place modification), because:

1. Native routed-experts capture avoids the per-layer device sync the old
   monkeypatch needed, so it likely does NOT inflate per_token_s_p50 the way
   the old approach would have -- but this hasn't been measured yet (see
   AXIS7.md#7's open items), so timing from this sweep is still not treated
   as axis6-comparable until that's confirmed cell-by-cell.
2. Default grid here is deliberately smaller than axis6's (see main()'s
   defaults): same steps x topk axis, one representative (B, rate) cell
   each. CORRECTION (AXIS7.md#16, take-3 sweep): the original reasoning
   here -- "expert routing is a property of WHICH TOKENS the model draws
   and HOW THE DRAFT TREE branches, not of queueing/admission load" -- is
   WRONG. Verified directly on the take-3 (verify_batch_expert_hooks.py)
   sweep: expert-footprint counts move substantially with B/rate, driven
   by verify-batch coalescing under continuous batching (a higher B means
   more concurrent requests get batched into one observed verify step).
   Left as a single-load-point default here anyway since this take-2
   sweep's signal (accepted-token-only, AXIS7.md#4) was already superseded
   before the load-dependence check was run -- not worth re-verifying on a
   signal no longer treated as primary.

Reuses launch/run machinery from sweep_sglang_depth_width.py rather than
duplicating it -- only the --enable-return-routed-experts flag, the
per-request return_routed_experts field, and the decode/aggregation step
are new.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time

import requests

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)
# scripts/ has no __init__.py (no other script in this repo imports across
# scripts/, all are standalone) -- import by module name off scripts/ on
# sys.path directly rather than introducing a `scripts.` package that
# nothing else here uses.
sys.path.insert(0, _SCRIPTS_DIR)
from sweep_sglang_depth_width import filter_overlong, stop_server, wait_for_server
from specloop_rt.workload import homogeneous

HF_HOME = "/root/spec_decode_env/hf_cache"


def launch_server(model_path: str, draft_path: str, num_steps: int, topk: int,
                   num_draft_tokens: int, context_length: int, max_running_requests: int,
                   port: int, log_path: str, dtype: str = "bfloat16") -> subprocess.Popen:
    """Same as sweep_sglang_depth_width.launch_server plus
    --enable-return-routed-experts (server_args.py) -- the flag that turns
    on the native capture buffer this sweep reads from (see module
    docstring). Not reused from sweep_sglang_depth_width.py directly since
    that function doesn't take extra flags and axis6 must never carry this
    flag (keeps axis6's grid uncontaminated by any capture overhead, however
    small -- see module docstring point 1)."""
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
        "--enable-return-routed-experts",
    ]
    logf = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)


def send_one_with_experts(port: int, prompt: str, max_new_tokens: int,
                           arrival_s: float = None) -> dict:
    """Same shape as sweep_sglang_depth_width.send_one plus
    return_routed_experts=True on the request and the decoded routed_experts
    array in the result. return_routed_experts is a top-level GenerateReqInput
    field (io_struct.py), NOT nested in sampling_params -- confirmed live
    (a first attempt nesting it under sampling_params silently returned no
    routed_experts key at all, no error)."""
    submit_wall = time.monotonic()
    r = requests.post(
        f"http://localhost:{port}/generate",
        json={"text": prompt,
              "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
              "return_routed_experts": True},
        timeout=180,
    )
    wall = time.monotonic() - submit_wall
    r.raise_for_status()
    d = r.json()
    mi = d["meta_info"]
    routed_experts_b64 = mi.get("routed_experts")
    routed_experts = None
    if routed_experts_b64:
        import numpy as np
        routed_experts = np.frombuffer(base64.b64decode(routed_experts_b64), dtype=np.int32)
    return {
        "wall_s": wall,
        "arrival_s": arrival_s,
        "e2e_latency": mi.get("e2e_latency"),
        "completion_tokens": mi.get("completion_tokens"),
        "spec_accept_rate": mi.get("spec_accept_rate"),
        "spec_accept_length": mi.get("spec_accept_length"),
        "routed_experts": routed_experts,
    }


def run_open_loop_with_experts(port: int, trace: list, max_new_tokens: int) -> list:
    """Same pacing as sweep_sglang_depth_width.run_open_loop (open-loop,
    paced to trace[i].arrival_s) -- duplicated rather than imported because
    it must call send_one_with_experts, not send_one."""
    import concurrent.futures
    t0 = time.monotonic()
    results = [None] * len(trace)

    def submit_one(i, req):
        dt = req.arrival_s - (time.monotonic() - t0)
        if dt > 0:
            time.sleep(dt)
        n_tok = min(req.max_tokens, max_new_tokens)
        results[i] = send_one_with_experts(port, req.prompt, n_tok, arrival_s=req.arrival_s)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(trace))) as ex:
        futs = [ex.submit(submit_one, i, req) for i, req in enumerate(trace)]
        for f in futs:
            f.result()
    return [r for r in results if r is not None]


def summarize_with_footprint(rows: list, num_layers: int, topk_size: int) -> dict:
    """Timing/acceptance summary (same fields as
    sweep_sglang_depth_width.summarize) plus expert-footprint stats decoded
    from each row's routed_experts array.

    SCOPE, READ BEFORE TRUSTING THESE NUMBERS AS A TEST OF THE DESIGN
    BRIEF'S HYPOTHESIS (see AXIS7.md#7 for the full trace): the native
    routed_experts capture is TARGET-MODEL, ACCEPTED-TOKEN-ONLY. Confirmed
    live: a request with prompt_tokens=22, completion_tokens=64 returned a
    routed_experts buffer of exactly (22+64-1)=85 tokens' worth of routing
    -- one row per token that received a permanent KV-cache slot, not one
    row per verify-batch candidate. capture_routed_experts_if_allowed() DOES
    fire for every draft candidate during the verify forward
    (unconditionally, inside _post_process_topk_ids), but the per-request
    response only reads back the device buffer at out_cache_loc (the
    KV-cache slot index) -- and rejected draft tokens never get a permanent
    KV slot, so their routing is captured on-device for an instant and then
    never surfaces here. There is no client-facing way to recover it; doing
    so would need a further SGLang-internals patch reading the pre-finalize
    device buffer, out of scope for this pass (see AXIS7.md#7's decision).

    CONSEQUENCE FOR WHAT distinct_experts MEANS: computed per single token
    per layer, distinct-experts is trivially topk_size (a token's own top-k
    is definitionally topk_size distinct experts -- confirmed on live data,
    which returned exactly 8.0 with a naive per-token version of this
    function; that version is what this docstring replaces). The stat below
    is instead aggregated PER REQUEST PER LAYER, across every accepted
    token in that request -- i.e. "how many distinct experts, and how
    imbalanced, did this request's whole accepted continuation touch at
    this layer" -- which is a real, if narrower-than-hoped, signal: it can
    still show whether a wider/deeper tree changes which or how many
    experts the SURVIVING continuation exercises, just not the rejected-
    draft cost contribution the design brief's cost model wants directly.

    Each row's flat routed_experts array is num_accepted_tokens_in_request *
    num_layers * topk_size int32s -- derive num_accepted_tokens from array
    length // (num_layers * topk_size) rather than assuming it equals
    completion_tokens (prefill contributes one row too)."""
    import numpy as np
    if not rows:
        return {"n": 0}
    wall = np.array([r["wall_s"] for r in rows])
    tok = np.array([r["completion_tokens"] for r in rows if r["completion_tokens"]])
    accept_rate = [r["spec_accept_rate"] for r in rows if r["spec_accept_rate"] is not None]
    accept_len = [r["spec_accept_length"] for r in rows if r["spec_accept_length"] is not None]
    per_token_s = wall / np.maximum(tok, 1) if len(tok) == len(wall) else wall

    per_row_shape = num_layers * topk_size
    distinct_experts, max_per_expert, tokens_routed = [], [], []
    for r in rows:
        arr = r.get("routed_experts")
        if arr is None or arr.size == 0 or arr.size % per_row_shape != 0:
            continue
        n_tokens = arr.size // per_row_shape
        mat = arr.reshape(n_tokens, num_layers, topk_size)
        for layer_id in range(num_layers):
            # aggregate ACROSS all accepted tokens in this request, at this
            # layer -- not per single token (see docstring: per-token
            # distinct-experts is trivially topk_size, not a useful signal)
            experts = mat[:, layer_id, :].reshape(-1)
            experts = experts[experts >= 0]
            if experts.size == 0:
                continue
            vals, counts = np.unique(experts, return_counts=True)
            distinct_experts.append(len(vals))
            max_per_expert.append(int(counts.max()))
            tokens_routed.append(int(experts.size))

    footprint = {}
    if distinct_experts:
        de = np.array(distinct_experts, dtype=float)
        mpe = np.array(max_per_expert, dtype=float)
        tr = np.array(tokens_routed, dtype=float)
        imbalance = mpe / np.maximum(tr / np.maximum(de, 1), 1e-9)
        footprint = {
            "moe_layer_calls": int(len(distinct_experts)),
            "mean_distinct_experts": float(de.mean()),
            "p95_distinct_experts": float(np.percentile(de, 95)),
            "mean_max_tokens_per_expert": float(mpe.mean()),
            "mean_imbalance": float(imbalance.mean()),
        }

    return {
        "n": len(rows),
        "wall_p50": float(np.percentile(wall, 50)),
        "wall_p95": float(np.percentile(wall, 95)),
        "per_token_s_p50": float(np.percentile(per_token_s, 50)),
        "mean_accept_rate": float(np.mean(accept_rate)) if accept_rate else None,
        "mean_accept_length": float(np.mean(accept_len)) if accept_len else None,
        **footprint,
    }


def run_cell(model_path, draft_path, num_steps, topk, num_draft_tokens,
             context_length, batch, rate, duration, rtype, max_new_tokens,
             port, out_dir, tag, num_layers, topk_size, dtype="bfloat16"):
    effective_draft_tokens = min(num_draft_tokens, num_steps * topk + 1)
    log_path = os.path.join(out_dir, f"server_{tag}.log")
    proc = launch_server(model_path, draft_path, num_steps, topk, effective_draft_tokens,
                         context_length, batch, port, log_path, dtype=dtype)
    try:
        wait_for_server(port, proc)
        trace = homogeneous(rtype=rtype, rate=rate, duration=duration, seed=0,
                            use_real_corpus=True)
        trace = filter_overlong(trace, context_length, max_new_tokens)
        send_one_with_experts(port, trace[0].prompt, 8)
        rows = run_open_loop_with_experts(port, trace, max_new_tokens)
        summary = summarize_with_footprint(rows, num_layers, topk_size)
    finally:
        stop_server(proc, port)
    if "mean_distinct_experts" not in summary:
        print(f"WARNING: {tag} produced no expert-activation rows "
              f"(check server_{tag}.log; confirm --enable-return-routed-experts "
              f"took effect and num_layers/topk_size match the model)",
              flush=True)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser("SGLang expert-activation footprint sweep (Axis-7 C1)")
    p.add_argument("--model-path", required=True)
    p.add_argument("--draft-path", required=True)
    p.add_argument("--num-layers", type=int, required=True,
                   help="target model's num_hidden_layers (Qwen3-30B-A3B-Instruct-2507: 48) "
                        "-- needed to reshape the flat routed_experts buffer; check the "
                        "model's config.json rather than assume this default is still current")
    p.add_argument("--topk-size", type=int, required=True,
                   help="target model's num_experts_per_tok (Qwen3-30B-A3B-Instruct-2507: 8)")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--topks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-draft-tokens", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--Bs", type=int, nargs="+", default=[16],
                   help="one representative batch cap by default -- see module "
                        "docstring for why this sweep doesn't need axis6's full B axis")
    p.add_argument("--rates", type=float, nargs="+", default=[4.0])
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--rtype", default="code")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--port", type=int, default=30020)
    p.add_argument("--out", default="results_gpu_sweep/sglang_expert_footprint_qwen3moe")
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

    print("\n=== mean_distinct_experts by (steps, topk) ===")
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
