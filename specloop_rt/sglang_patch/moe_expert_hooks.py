"""Expert-activation instrumentation for SGLang's Qwen3-MoE model (C1 in the
expert-aware tree-shaping design brief). The actual patch target is
TopK.forward (see install() below), not Qwen3MoeSparseMoeBlock.forward_normal
directly -- but the signal comes from the same call site, confirmed by
reading sglang==0.5.16 source directly
(python/sglang/srt/models/qwen3_moe.py):

  class Qwen3MoeSparseMoeBlock.forward_normal(self, hidden_states):
      router_logits, _ = self.gate(hidden_states)
      topk_output = self.topk(hidden_states, router_logits)   # <-- TopK.forward runs here
      final_hidden_states = self.experts(hidden_states, topk_output)

topk_output is a StandardTopKOutput(topk_weights, topk_ids, router_logits)
NamedTuple (python/sglang/srt/layers/moe/topk.py); topk_ids has shape
(num_tokens, top_k) and IS the per-token expert assignment for that layer,
that forward pass -- exactly the E(tree) signal the design brief's cost
model needs, with no reimplementation of routing logic required.

WHY A MONKEYPATCH, NOT AN SGLANG SOURCE EDIT: this repo already has that
convention (specloop_rt/vllm_patch/) for the same reason -- pip-installed
sglang gets reinstalled/upgraded independent of this repo, and a source
edit would silently vanish or conflict on the next `pip install sglang`.

COST OF THIS HOOK: topk_ids.to("cpu") forces a device sync at every MoE
layer of every verify step -- on a model with dozens of MoE layers this
measurably inflates per_token_s_p50, the very latency axis6's cost model is
built on. That means C1 runs are a SEPARATE sweep from axis6's timed runs,
never both at once (see AXIS7.md#2) -- this hook trades timing fidelity for
routing visibility on purpose, it does not try to have both for free.
`non_blocking=True` is kept on the .to() call only because it is free when
true async transfer isn't possible (falls back to sync silently) and cheap
insurance for any future host-pinned-memory build of this hook; do not read
it as removing the sync cost above.

ACTIVATION: everything below is a no-op unless CAVEMAN_MOE_HOOK_OUT is set
in the process environment -- this module is imported unconditionally by
sitecustomize.py (see that file's docstring) in every sweep run, dense and
MoE alike, and must cost nothing when unused, especially on the dense
Llama runs where Qwen3MoeSparseMoeBlock is never even imported.
"""
from __future__ import annotations

import atexit
import json
import os
import threading

_ENV_OUT_PATH = "CAVEMAN_MOE_HOOK_OUT"
_ENV_CELL_TAG = "CAVEMAN_MOE_HOOK_CELL_TAG"

_lock = threading.Lock()
_buffer: list[dict] = []
_out_path: str | None = None
_cell_tag: str = ""
_installed = False


def _record(layer_id: int, num_tokens: int, topk_ids_cpu) -> None:
    # topk_ids_cpu: (num_tokens, top_k) int tensor, already off-device.
    # -1 marks a padded/invalid slot (see StandardTopKOutput usage in
    # sglang's fused-MoE dispatch) -- drop those before counting.
    flat = topk_ids_cpu.reshape(-1).tolist()
    experts = [e for e in flat if e >= 0]
    if not experts:
        return
    from collections import Counter
    counts = Counter(experts)
    with _lock:
        _buffer.append({
            "cell_tag": _cell_tag,
            "layer_id": layer_id,
            "num_tokens": num_tokens,
            "distinct_experts": len(counts),
            "max_tokens_per_expert": max(counts.values()),
            "tokens_routed": len(experts),
        })


def flush() -> None:
    """Append the buffer to CAVEMAN_MOE_HOOK_OUT as JSONL and clear it.
    _record() already pays its device-sync cost inline (see install()'s
    docstring) -- flush() only batches the resulting CPU-side dicts into
    fewer, larger file writes. Safe to call repeatedly (e.g. once per
    request) -- each call only writes what accumulated since the last
    flush, so a crash mid-sweep loses at most one request's worth of rows,
    not the whole cell."""
    global _buffer
    if _out_path is None:
        return
    with _lock:
        rows, _buffer = _buffer, []
    if not rows:
        return
    with open(_out_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def install() -> None:
    """Monkeypatch TopK.forward (python/sglang/srt/layers/moe/topk.py),
    NOT Qwen3MoeSparseMoeBlock.forward_normal. Wrapping the routing call
    itself, rather than the surrounding block method, means this hook never
    has to re-derive forward_normal's post-experts all-reduce/reshape tail
    (ep_size/tp_size branches, view() reshape) -- that logic stays exactly
    sglang's own, untouched, so a version bump to those branches can't
    silently desync this hook from correct behavior. The only thing
    intercepted is the return value of routing, which is already the
    complete signal this hook needs (topk_ids).

    self.topk is a per-block TopK module instance (one per MoE layer), so
    layer_id is read from the bound TopK instance itself, not threaded in.

    Idempotent -- safe to call multiple times (mirrors sglang's own
    plugin-loading idempotency convention, srt/plugins/__init__.py)."""
    global _out_path, _cell_tag, _installed
    _out_path = os.environ.get(_ENV_OUT_PATH)
    _cell_tag = os.environ.get(_ENV_CELL_TAG, "")
    if not _out_path:
        return  # inert: no env var set, e.g. every dense/Llama run
    if _installed:
        return

    try:
        from sglang.srt.layers.moe.topk import TopK
    except ImportError:
        return  # sglang not on this process's import path at all

    orig_call = TopK.forward

    def patched_forward(self, hidden_states, router_logits, *args, **kwargs):
        topk_output = orig_call(self, hidden_states, router_logits, *args, **kwargs)
        try:
            topk_ids = topk_output.topk_ids
        except AttributeError:
            # BypassedTopKOutput / PackedTopKOutput / TritonKernelTopKOutput
            # variants (topk.py) don't expose a plain topk_ids tensor the
            # same way -- skip recording rather than guess a shape.
            return topk_output
        num_tokens = hidden_states.shape[0]
        topk_ids_cpu = topk_ids.to("cpu", non_blocking=True)
        layer_id = getattr(self, "layer_id", -1)
        _record(layer_id, num_tokens, topk_ids_cpu)
        return topk_output

    TopK.forward = patched_forward
    TopK._forward_unpatched = orig_call
    _installed = True
    atexit.register(flush)
