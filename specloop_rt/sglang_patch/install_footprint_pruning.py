"""Live installer for footprint-aware draft pruning (idea 1), wiring
specloop_rt.sglang_patch.footprint_aware_pruning's graph-safe
footprint_aware_select_top_k_tokens into a running SGLang server via the
same sitecustomize.py/PYTHONPATH injection convention as
verify_batch_expert_hooks.py (see that module and AXIS7.md#10 for why this
injection point, not sglang's own plugin framework).

VERSION NOTE: this repo's SGLang 0.5.17 environment has a real,
upstream-confirmed A100 (sm80) incompatibility -- sgl-kernel==0.4.5 (the
version 0.5.17 hard-pins) ships only sm90/sm100 compiled kernels, verified
by inspecting the actual wheel contents, not a local misconfiguration. This
hook therefore targets sglang==0.4.10 (pins sgl-kernel==0.2.8, confirmed to
still ship the universal/sm80-compatible common_ops.abi3.so), the version
this session's live GPU test actually runs against. select_top_k_tokens
lives in a different module here than in 0.5.17
(sglang.srt.speculative.eagle_utils, not .spec_utils) -- confirmed by
reading the installed 0.4.10 source directly, not assumed by version-number
proximity.

WHERE TO PATCH, THE PART THAT ACTUALLY MATTERS: eagle_worker.py does
`from sglang.srt.speculative.eagle_utils import select_top_k_tokens`,
binding a local name in eagle_worker's OWN module namespace at import time.
Patching sglang.srt.speculative.eagle_utils.select_top_k_tokens after
eagle_worker has already imported it does NOT change what
eagle_worker.draft_forward() calls -- Python's `from X import Y` binds a
new reference, it does not alias X.Y live. This is the same class of
mistake take-1's hook made in a different way (patching a class whose
bound method reference had already been captured elsewhere) -- see
moe_expert_hooks.py's docstring. Must patch the name in
sglang.srt.speculative.eagle_worker's OWN namespace, not eagle_utils's.

CUDA-graph safety: confirmed (this session, both by reading source and by
an isolated torch.cuda.graph capture/replay test -- see
footprint_aware_pruning.py's STATUS section) that
eagle_draft_cuda_graph_runner.py's capture_one_batch_size calls
self.eagle_worker.draft_forward(forward_batch) in eager Python, once, at
capture time -- so a patch here fires and its tensor ops get captured
into the graph. The patched function must therefore stay pure-tensor-ops
(no Python-level loops/sets/.tolist() over draft-step-dependent data),
which footprint_aware_select_top_k_tokens already satisfies by
construction (see that function's own docstring).

LAZY INITIALIZATION -- root cause of an earlier goodput regression, fixed
here: sitecustomize.py (this package's injection shim) runs install()
UNCONDITIONALLY in every SGLang subprocess -- scheduler, detokenizer, AND
several torch._inductor.compile_worker processes (confirmed live: sglang
spawns 8 total for this server config, per-process GPU memory checked via
`nvidia-smi --query-compute-apps`). Only the SCHEDULER process ever calls
select_top_k_tokens; the other ~7 never do. The original version of this
file loaded the checkpoint and built RoutingProxyHead().cuda() eagerly,
inside install() itself -- meaning ALL 8 subprocesses independently paid
the cost of a CUDA context + proxy construction, even the 7 that would
never use it. Measured live: ~588MB GPU memory per non-scheduler
subprocess (mostly CUDA context/allocator overhead, not the proxy's own
~25-50MB of weights) x 7 idle subprocesses = ~4.1GB of GPU memory
consumed for nothing, directly starving the scheduler's own KV-cache
budget (max_total_num_tokens dropped ~20% in the goodput comparison that
first surfaced this). Root-caused by adding temporary per-call memory
tracing (CAVEMAN_PRUNING_DEBUG_MEM) and comparing against
`nvidia-smi --query-compute-apps=pid,used_memory` -- the memory was NOT
concentrated in the scheduler process's own capture-time deltas (measured
directly: ~40MB total across the whole capture phase), ruling out the
pruning function's own tensor ops as the cause and pointing at
per-process eager initialization instead.

FIX: install() now only patches a thin wrapper unconditionally (cheap --
no CUDA, no torch.load) in every subprocess; the actual checkpoint load /
RoutingProxyHead().cuda() / covered_mask allocation is deferred to the
FIRST REAL CALL of the patched function, inside _lazy_init(). Since only
the scheduler subprocess ever calls select_top_k_tokens, only the
scheduler ever triggers _lazy_init() -- the other subprocesses' patched
wrapper is simply never invoked, and never pays the GPU-memory cost.

ACTIVATION: no-op unless CAVEMAN_PRUNING_HOOK_OUT is set (loosely mirrors
verify_batch_expert_hooks.py's env-gating convention, reusing the CAVEMAN_
prefix for consistency across this repo's sglang_patch hooks) --
CAVEMAN_PRUNING_CKPT_PATH must point at a routing_proxy_head.pt checkpoint
(see scripts/train_routing_proxy.py) and CAVEMAN_PRUNING_LAMBDA sets the
cost/value tradeoff (default 1.0, unfit -- see footprint_aware_pruning.py's
STATUS section).
"""
from __future__ import annotations

import os
import sys

_ENV_ENABLE = "CAVEMAN_PRUNING_HOOK_OUT"
_ENV_CKPT = "CAVEMAN_PRUNING_CKPT_PATH"
_ENV_LAMBDA = "CAVEMAN_PRUNING_LAMBDA"

_installed = False
_call_count = 0
_last_error = None

# Lazy-init state, populated on first real call in whichever subprocess
# actually calls select_top_k_tokens (the scheduler, in practice) -- see
# module docstring for why this must NOT happen eagerly in install().
_lazy = {
    "proxy": None,
    "covered_mask": None,
    "proxy_topk": None,
    "lam": None,
    "orig_select_top_k_tokens": None,
    "failed": False,
}


def _lazy_init(eagle_worker) -> bool:
    """Loads the checkpoint and builds the proxy/coverage buffer, once,
    in whichever process calls this first. Returns True on success, False
    if init failed (checkpoint missing, etc.) -- callers fall back to the
    unpatched original on False, same failure-handling convention as the
    per-call exception handler below."""
    if _lazy["proxy"] is not None or _lazy["failed"]:
        return _lazy["proxy"] is not None

    import torch

    ckpt_path = os.environ.get(_ENV_CKPT)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[caveman-pruning] CAVEMAN_PRUNING_CKPT_PATH={ckpt_path!r} "
              f"missing or not found -- hook NOT activated in this process. "
              f"Train one with scripts/train_routing_proxy.py first.",
              file=sys.stderr, flush=True)
        _lazy["failed"] = True
        return False

    from specloop_rt.sglang_patch.footprint_aware_pruning import RoutingProxyHead

    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    proxy = RoutingProxyHead(
        ckpt["draft_hidden_size"], ckpt["num_experts"], ckpt["num_target_layers"],
    ).cuda()
    proxy.load_state_dict(ckpt["state_dict"])
    # Trained in float32 (scripts/train_routing_proxy.py never casts), but
    # eagle_worker's real hidden_states arrive as bfloat16 (matching the
    # server's --dtype) -- cast the proxy to match, confirmed necessary by
    # a live RuntimeError (mat1/mat2 dtype mismatch) on the first GPU
    # integration attempt, not assumed preemptively.
    proxy = proxy.to(torch.bfloat16)
    proxy.eval()

    num_target_layers = ckpt["num_target_layers"]
    num_experts = ckpt["num_experts"]

    # Persistent, per-REQUEST coverage buffer: sglang batches multiple
    # concurrent requests, each growing its OWN draft tree, and
    # hidden_states at a call is (true_batch_size * topk, hidden_size), not
    # one flat candidate list for a single tree (confirmed live: a first
    # integration attempt using one shared (num_layers, num_experts)
    # buffer threw shape-mismatch errors immediately, and would have
    # silently conflated coverage across unrelated requests even where
    # shapes happened to align). Sized generously against the server's
    # configured --max-running-requests rather than reading it from the
    # live batch (a CUDA-graph-captured buffer must have a fixed
    # allocation size decided before capture, not derived per-call).
    _max_batch_size = int(os.environ.get("CAVEMAN_PRUNING_MAX_BATCH", "64"))
    covered_mask = torch.zeros(_max_batch_size, num_target_layers, num_experts, device="cuda")

    _lazy["proxy"] = proxy
    _lazy["covered_mask"] = covered_mask
    _lazy["proxy_topk"] = ckpt["topk"]
    _lazy["lam"] = float(os.environ.get(_ENV_LAMBDA, "1.0"))
    _lazy["orig_select_top_k_tokens"] = eagle_worker.select_top_k_tokens_orig

    print(f"[caveman-pruning] lazily initialized in pid={os.getpid()} "
          f"(checkpoint={ckpt_path}, lambda={_lazy['lam']})", file=sys.stderr, flush=True)
    return True


def install() -> None:
    """Idempotent -- safe to call multiple times. No-ops unless
    CAVEMAN_PRUNING_HOOK_OUT is set, same convention as this package's
    other hooks. Cheap in every subprocess (no CUDA, no torch.load) -- see
    module docstring for why the actual GPU initialization is deferred to
    _lazy_init(), triggered only in whichever process makes the first
    real call."""
    global _installed
    if not os.environ.get(_ENV_ENABLE):
        return
    if _installed:
        return

    try:
        from sglang.srt.speculative import eagle_worker
    except ImportError as e:
        print(f"[caveman-pruning] ImportError loading eagle_worker: {e!r} "
              f"-- hook NOT installed", file=sys.stderr, flush=True)
        return

    from specloop_rt.sglang_patch.footprint_aware_pruning import compute_footprint_adjusted_topk_p

    eagle_worker.select_top_k_tokens_orig = eagle_worker.select_top_k_tokens

    def patched_select_top_k_tokens(i, topk_p, topk_index, hidden_states, scores, topk):
        global _call_count, _last_error
        _call_count += 1

        if not _lazy_init(eagle_worker):
            return eagle_worker.select_top_k_tokens_orig(i, topk_p, topk_index, hidden_states, scores, topk)

        proxy = _lazy["proxy"]
        covered_mask = _lazy["covered_mask"]
        proxy_topk = _lazy["proxy_topk"]
        lam = _lazy["lam"]
        orig_select_top_k_tokens = _lazy["orig_select_top_k_tokens"]

        if i == 0:
            # KNOWN GAP, not silently papered over: this zeros slots
            # [0, b) on every fresh-tree call, assuming request-to-slot
            # assignment is stable and starts at 0 -- but
            # select_top_k_tokens's signature exposes no request IDENTITY
            # (only tensors), so this hook cannot actually confirm which
            # buffer slots belong to which real request across calls, or
            # whether a fresh-tree request always lands at low slot
            # indices when OTHER requests are simultaneously mid-tree
            # (i>0) in the same batch. Coverage semantics are verified
            # correct for the single-fresh-batch case this session's live
            # test exercises (one request/small batch at a time); doing
            # this correctly under real concurrent mixed-step traffic
            # needs request-id-keyed coverage, which requires plumbing an
            # identity signal into this hook from a call site with more
            # context than select_top_k_tokens's own signature carries
            # (e.g. forward_batch.req_pool_indices, available one level up
            # in eagle_worker.draft_forward, not here) -- not done this
            # session; flagged rather than assumed correct.
            b = hidden_states.shape[0] // topk
            covered_mask[:b].zero_()
        try:
            adjusted_topk_p = compute_footprint_adjusted_topk_p(
                i=i, topk_p=topk_p, hidden_states=hidden_states, topk=topk,
                routing_proxy=proxy, covered_mask=covered_mask,
                proxy_topk=proxy_topk, lam=lam,
            )
        except Exception as e:
            # Never let a research hook take down real serving -- but DO
            # surface the failure loudly (same lesson AXIS7.md#10 already
            # learned from a silent except: pass hiding a real bug).
            _last_error = repr(e)
            import traceback
            print(f"[caveman-pruning] EXCEPTION in patched_select_top_k_tokens: {e!r}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return orig_select_top_k_tokens(i, topk_p, topk_index, hidden_states, scores, topk)
        # Feed the ADJUSTED topk_p into the REAL, unmodified
        # select_top_k_tokens -- sglang's own tree-topology bookkeeping
        # (tree_info's parent-pointer encoding, consumed downstream by
        # verify-step attention masking) is derived from topk_p/scores via
        # its own torch.topk/fast_topk calls; perturbing the ranking
        # SIGNAL and letting that already-correct code do the actual
        # selection avoids reimplementing (and risking a silent bug in)
        # the tree-index math itself. See compute_footprint_adjusted_
        # topk_p's docstring for why this approach was chosen over a
        # from-scratch reimplementation, and its lam=0-identity /
        # i==0-passthrough guarantees (both verified by a standalone test
        # before this was wired in).
        return orig_select_top_k_tokens(i, adjusted_topk_p, topk_index, hidden_states, scores, topk)

    eagle_worker.select_top_k_tokens = patched_select_top_k_tokens
    _installed = True
    print(f"[caveman-pruning] patch installed (lazy) in pid={os.getpid()}", file=sys.stderr, flush=True)
