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
    return


def install(num_layers: int, topk_size: int) -> None:
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
            return output
        try:
            topk_gpu = capture_output.topk
            num_tokens = topk_gpu.shape[0]
            topk_cpu = topk_gpu[:, :, :topk_size].to("cpu", non_blocking=True)
            _record(num_tokens, num_layers, topk_size, topk_cpu)
        except Exception as e:
            import sys, traceback
            print(f"[caveman-canary] verify hook EXCEPTION: {e!r}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
        return output

    ModelRunner.forward = patched_forward
    ModelRunner._forward_unpatched = orig_forward
    _installed = True
