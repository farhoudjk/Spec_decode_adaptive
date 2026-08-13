"""C1, take 3: capture the FULL verify-batch expert routing (proposed draft
tokens, accepted AND rejected), not just the accepted-token-only signal
`--enable-return-routed-experts` exposes natively. See AXIS7.md#10 for the
full trace of why this exists and what the prior two attempts got wrong.

WHERE THE REJECTED-TOKEN DATA ACTUALLY GOES (confirmed by reading sglang
0.5.17 source directly): `capture_routed_experts_if_allowed()`
(layers/moe/topk.py) fires unconditionally inside `_post_process_topk_ids`,
for every TopK.forward call including the full verify batch -- so the
routing data for rejected candidates genuinely gets written into
`RoutedExpertsCapturer.device_cache.buffer` (state_capturer/base.py) during
every verify step. It is NOT lost at capture time. It is lost one step
later: `ModelRunner.forward()` (model_executor/model_runner.py) calls
`experts_capturer.on_forward_end(forward_batch, ...)` once per forward,
which returns a `TopkCaptureOutput` holding `topk` (device tensor, full
verify-batch shape) and `out_cache_loc` (also verify-batch-sized -- every
proposed draft token gets a provisional KV slot reserved just to compute
its verify logits, whether or not it's ultimately kept). This
`TopkCaptureOutput` is NOT finalized inside `forward()` -- `.finalize()` is
called later, from `batch_result_processor.py`
(`result.routed_experts_output.finalize()`), which does
`host_cache.buffer[out_cache_loc] = topk` -- writing into a cache indexed
by KV-cache slot. The client-facing read path, `get_topk()`
(state_capturer/base.py), then walks `req_to_token_pool.req_to_token[...]`
-- the request's COMMITTED, POST-REJECTION-SAMPLING token sequence -- so
only entries at still-live KV slots are ever read back out. Rejected
tokens' slots get freed/reused after rejection sampling, so their routing
data, though genuinely written, is never read by anything downstream. The
data isn't destroyed, it's just structurally unreachable via the public
`get_topk()`/`--enable-return-routed-experts` path.

THE FIX: intercept `ModelRunner.forward()`'s return value BEFORE
`finalize()` runs, while `output.routed_experts_output.topk` and
`.out_cache_loc` still describe the full verify batch. Only meaningful
during a verify forward (`forward_batch.forward_mode.is_target_verify()`)
-- decode/prefill/draft-extend forwards have exactly one candidate per
position already (no rejection to lose data to), so this hook only records
during verify steps, which is also exactly the step the design brief's
cost hypothesis is about.

WHY THIS PATCH POINT IS GRAPH-SAFE (unlike the take-1 monkeypatch on
TopK.forward, see moe_expert_hooks.py's docstring for that failure):
`ModelRunner.forward()`'s CUDA graph replay happens INSIDE `_forward_raw()`
(`if can_run_graph: ret = self.decode_cuda_graph_runner.execute(...); return`)
and returns before `on_forward_end()` is ever called -- `on_forward_end()`
runs in `forward()`, strictly AFTER `_forward_raw()` returns, i.e. after
graph replay has already completed for that step, in normal eager Python.
This is the same reason the native `--enable-return-routed-experts`
mechanism itself works at all despite CUDA graphs: the on-device `capture()`
write happens INSIDE the graph (a plain buffer write, which the graph
records fine), but the Python-level `TopkCaptureOutput` orchestration
around it runs outside the graph every step, patchable like any normal
Python call.

WHY PATCH `ModelRunner.forward`, NOT `on_forward_end` DIRECTLY: the
capturer's `on_forward_end()` call is inline inside `forward()`'s large
body (not a separately overridable seam), so there's no smaller hook point
without editing sglang source. Wrapping the whole `forward()` method and
reading its return value is the smallest change that reaches the data
before `finalize()` consumes it.

ACTIVATION: no-op unless CAVEMAN_VERIFY_HOOK_OUT is set, same convention
as moe_expert_hooks.py. Both hook modules can coexist on PYTHONPATH (they
patch different classes/methods) but only this one is used by
sweep_sglang_verify_footprint.py.
"""
from __future__ import annotations

import json
import os
import threading

_ENV_OUT_PATH = "CAVEMAN_VERIFY_HOOK_OUT"
_ENV_CELL_TAG = "CAVEMAN_VERIFY_HOOK_CELL_TAG"

_lock = threading.Lock()
_out_path: str | None = None
_cell_tag: str = ""
_installed = False


def _record(num_tokens: int, num_layers: int, topk_size: int, topk_cpu) -> None:
    # topk_cpu: (num_tokens, num_layers, topk_size) int tensor, off-device.
    # One row per record = one verify-forward call (i.e. one MoE-layer's
    # full candidate-token footprint for that verify step), matching the
    # design brief's E(tree) unit -- NOT one row per token, unlike take-2's
    # bug (see AXIS7.md#5 for why per-token aggregation was wrong).
    #
    # WRITE-THROUGH, NOT BUFFER-AND-ATEXIT: take-1's moe_expert_hooks.py
    # buffered rows and relied on atexit.register(flush) to write them at
    # process shutdown -- untested there (it never recorded a row in the
    # first place, see AXIS7.md#2) and turned out to be WRONG here: the
    # sweep scripts' stop_server() sends SIGTERM to the process group, and
    # a raw SIGTERM does NOT run atexit handlers unless the receiving
    # process installs its own handler that calls sys.exit() -- confirmed
    # empirically: canaries showed real routed_experts_output objects with
    # correct 13-token (num_draft_tokens-sized) out_cache_loc arriving at
    # patched_forward, but the buffered rows never reached disk because
    # the server process's SIGTERM-triggered shutdown never ran the
    # registered atexit callback. Fixed by writing every record straight
    # to disk immediately instead of buffering -- costs an open+write+close
    # per verify step (this sweep already isn't latency-comparable to
    # axis6, see module docstring), but guarantees no data loss regardless
    # of how the server process is torn down.
    from collections import Counter
    rows = []
    for layer_id in range(num_layers):
        flat = topk_cpu[:, layer_id, :].reshape(-1).tolist()
        experts = [e for e in flat if e >= 0]
        if not experts:
            continue
        counts = Counter(experts)
        rows.append({
            "cell_tag": _cell_tag,
            "layer_id": layer_id,
            "verify_batch_tokens": num_tokens,
            "distinct_experts": len(counts),
            "max_tokens_per_expert": max(counts.values()),
            "tokens_routed": len(experts),
        })
    if rows and _out_path is not None:
        with _lock:
            with open(_out_path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")


def flush() -> None:
    """Kept as a no-op-safe API for callers that still invoke it (e.g. any
    atexit registration left in place) -- _record() is write-through now,
    so there is nothing left to flush. Not removed to avoid an AttributeError
    in any code still calling verify_batch_expert_hooks.flush()."""
    return


def install(num_layers: int, topk_size: int) -> None:
    """Monkeypatch ModelRunner.forward. num_layers/topk_size must be passed
    explicitly (same requirement as sweep_sglang_expert_footprint.py's
    --num-layers/--topk-size) -- there is no cheap way to introspect these
    from inside the patch without importing the model config, and the
    caller already has them from the launch config.

    Idempotent -- safe to call multiple times."""
    global _out_path, _cell_tag, _installed
    _out_path = os.environ.get(_ENV_OUT_PATH)
    _cell_tag = os.environ.get(_ENV_CELL_TAG, "")
    if not _out_path:
        return
    if _installed:
        return

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except ImportError:
        return

    orig_forward = ModelRunner.forward

    def patched_forward(self, forward_batch, *args, **kwargs):
        output = orig_forward(self, forward_batch, *args, **kwargs)
        try:
            is_verify = forward_batch.forward_mode.is_target_verify()
        except Exception:
            is_verify = False
        if not is_verify:
            return output
        capture_output = getattr(output, "routed_experts_output", None)
        if capture_output is None:
            # no_copy_to_cpu=False path (disable_overlap_schedule) already
            # wrote straight to host_cache and returned None -- the
            # verify-batch data is already gone by the time we see it here.
            # Only the no_copy_to_cpu=True (overlap-scheduling, the
            # default) path gives us a TopkCaptureOutput to intercept.
            # Confirmed live: default config gives real TopkCaptureOutput
            # objects here with verify-batch-sized (num_draft_tokens,
            # not 1) out_cache_loc arrays -- see AXIS7.md#10.
            return output
        try:
            topk_gpu = capture_output.topk  # (num_verify_tokens, num_layers, device_topk_size)
            num_tokens = topk_gpu.shape[0]
            topk_cpu = topk_gpu[:, :, :topk_size].to("cpu", non_blocking=True)
            _record(num_tokens, num_layers, topk_size, topk_cpu)
        except Exception as e:
            # Never let instrumentation break real serving -- but DO
            # surface the failure, unlike a silent `except: pass`. A
            # silent swallow here is exactly what hid the real
            # write-through-vs-atexit bug (AXIS7.md#10) during initial
            # debugging; leave this loud.
            import sys, traceback
            print(f"[caveman-canary] verify hook EXCEPTION: {e!r}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
        return output

    ModelRunner.forward = patched_forward
    ModelRunner._forward_unpatched = orig_forward
    _installed = True
