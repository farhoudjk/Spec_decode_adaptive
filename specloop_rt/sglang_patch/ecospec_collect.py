from __future__ import annotations

import json
import os
import sys
import threading

_ENV_DRAFT_OUT = "CAVEMAN_ECOSPEC_DRAFT_OUT"
_ENV_VERIFY_OUT = "CAVEMAN_ECOSPEC_VERIFY_OUT"
_ENV_TOPK_SIZE = "CAVEMAN_ECOSPEC_TOPK_SIZE"

_installed_draft = False
_installed_verify = False
_lock = threading.Lock()


def _write_jsonl(path: str, row: dict) -> None:
    with _lock:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")


def _walk_ancestry_precut(parents_list_row_major, topk):
    import torch
    b = parents_list_row_major[0].shape[0]
    device = parents_list_row_major[0].device
    total_width = topk + sum(topk * topk for _ in range(len(parents_list_row_major) - 1))
    parent_of_flat = torch.full((b, total_width), -1, dtype=torch.long, device=device)

    global_offset = 0
    prev_survivors_flat = None
    for i, p in enumerate(parents_list_row_major):
        if i == 0:
            n = topk
            prev_survivors_flat = torch.arange(
                global_offset, global_offset + topk, device=device
            ).unsqueeze(0).expand(b, -1)
            global_offset += n
        else:
            n = topk * topk
            parent_local_survivor_slot = torch.arange(n, device=device) // topk
            gathered = prev_survivors_flat[:, parent_local_survivor_slot]
            parent_of_flat[:, global_offset:global_offset + n] = gathered
            prev_survivors_flat = p
            global_offset += n
    return parent_of_flat


def install_draft_side() -> bool:
    global _installed_draft
    out_path = os.environ.get(_ENV_DRAFT_OUT)
    if out_path is None:
        return False
    if _installed_draft:
        return True

    try:
        import torch
        from sglang.srt.speculative import eagle_worker_v2
    except ImportError as e:
        print(f"[caveman-ecospec] draft-side ImportError: {e!r}", file=sys.stderr, flush=True)
        return False

    if not hasattr(eagle_worker_v2, "organize_draft_results_orig"):
        eagle_worker_v2.organize_draft_results_orig = eagle_worker_v2.organize_draft_results

    orig_organize = eagle_worker_v2.organize_draft_results_orig
    _req_counter = {"n": 0}
    eagle_topk_env = os.environ.get("CAVEMAN_ECOSPEC_EAGLE_TOPK")
    eagle_topk = int(eagle_topk_env) if eagle_topk_env else None

    def patched_organize_draft_results(score_list, token_list, parents_list, num_draft_token):
        try:
            if eagle_topk is None:
                raise RuntimeError("CAVEMAN_ECOSPEC_EAGLE_TOPK not set")
            b = score_list[0].shape[0]
            flat_score = torch.cat(score_list, dim=1).flatten(1)
            total_width = flat_score.shape[1]

            parent_of_flat = _walk_ancestry_precut(parents_list, eagle_topk)

            result = orig_organize(score_list, token_list, parents_list, num_draft_token)
            parent_list, top_scores_index, draft_tokens = result

            req_id = _req_counter["n"]
            _req_counter["n"] += 1
            row = {
                "req_id": req_id,
                "b": b,
                "eagle_topk": eagle_topk,
                "total_width": total_width,
                "num_draft_token": int(num_draft_token),
                "flat_score": flat_score.tolist(),
                "parent_of_flat": parent_of_flat.tolist(),
                "top_scores_index": top_scores_index.tolist(),
            }
            _write_jsonl(out_path, row)
            return result
        except Exception as e:
            print(f"[caveman-ecospec] draft-side EXCEPTION: {e!r}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return orig_organize(score_list, token_list, parents_list, num_draft_token)

    eagle_worker_v2.organize_draft_results = patched_organize_draft_results
    _installed_draft = True
    print(f"[caveman-ecospec] draft-side collector installed, pid={os.getpid()}",
          file=sys.stderr, flush=True)
    return True


def install_verify_side() -> bool:
    global _installed_verify
    out_path = os.environ.get(_ENV_VERIFY_OUT)
    topk_size = os.environ.get(_ENV_TOPK_SIZE)
    if out_path is None or topk_size is None:
        return False
    if _installed_verify:
        return True
    topk_size = int(topk_size)

    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except ImportError as e:
        print(f"[caveman-ecospec] verify-side ImportError: {e!r}", file=sys.stderr, flush=True)
        return False

    if hasattr(ModelRunner, "_ecospec_forward_unpatched"):
        return True
    orig_forward = ModelRunner.forward
    ModelRunner._ecospec_forward_unpatched = orig_forward

    _step_counter = {"n": 0}

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
            topk_cpu = topk_gpu[:, :, :topk_size].to("cpu", non_blocking=True).tolist()
            num_tokens = topk_gpu.shape[0]

            req_tags = None
            try:
                req_pool_indices = forward_batch.req_pool_indices
                bs = req_pool_indices.shape[0]
                if num_tokens % bs == 0:
                    per_req = num_tokens // bs
                    req_tags = req_pool_indices.repeat_interleave(per_req).to("cpu").tolist()
                else:
                    print(f"[caveman-ecospec] num_tokens={num_tokens} not divisible by bs={bs}",
                          file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[caveman-ecospec] req_pool_indices unavailable: {e!r}",
                      file=sys.stderr, flush=True)

            step_id = _step_counter["n"]
            _step_counter["n"] += 1
            _write_jsonl(out_path, {
                "step_id": step_id,
                "topk_per_position": topk_cpu,
                "req_tag_per_position": req_tags,
            })
        except Exception as e:
            print(f"[caveman-ecospec] verify-side EXCEPTION: {e!r}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
        return output

    ModelRunner.forward = patched_forward
    _installed_verify = True
    print(f"[caveman-ecospec] verify-side collector installed, pid={os.getpid()}",
          file=sys.stderr, flush=True)
    return True


def install() -> None:
    install_draft_side()
    install_verify_side()
