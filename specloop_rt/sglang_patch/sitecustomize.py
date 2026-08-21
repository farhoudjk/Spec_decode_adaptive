"""Auto-import shim for this package's SGLang instrumentation hooks.

Python imports a module named `sitecustomize` at interpreter startup if it's
found on sys.path -- this is stdlib behavior (see site.py), not an SGLang
feature. Placing this file's directory on PYTHONPATH for a subprocess makes
it run before that subprocess's own __main__, with no code change to
sglang.launch_server's invocation needed.

WHY THIS IS THE RIGHT INJECTION POINT (not sglang's own plugin framework,
srt/plugins/__init__.py): that framework discovers plugins via setuptools
entry_points, which requires this hook to be a properly pip-installed
package -- overkill for instrumentation modules developed alongside one
experiment. It's also only invoked from launch_server.py's __main__ block
and a couple of engine entrypoints; sitecustomize.py is guaranteed to run
in EVERY subprocess that inherits PYTHONPATH, including the scheduler
subprocesses SGLang spawns via multiprocessing with start_method="spawn"
(confirmed by reading srt/entrypoints/engine.py) -- spawned processes are
fresh interpreters that re-read PYTHONPATH from the inherited environment,
so this shim reaches them without any per-process wiring.

PYTHONPATH NEEDS TWO ENTRIES, NOT ONE: this file itself must be directly on
sys.path for Python's site machinery to find a top-level `sitecustomize`
module at all (that lookup is flat, stdlib-defined, non-negotiable) -- so
this directory (specloop_rt/sglang_patch/) has to be on PYTHONPATH. But the
import below needs `specloop_rt` to resolve as a package, which requires
the REPO ROOT on PYTHONPATH too. The sweep script is responsible for
prepending both `<repo_root>` and `<repo_root>/specloop_rt/sglang_patch`
to PYTHONPATH in the launched server's env.

THREE INDEPENDENT HOOKS LIVE HERE, all installed unconditionally below, all
individually env-gated so any subset can be active per-process:
- moe_expert_hooks.py (CAVEMAN_MOE_HOOK_OUT): take-1, patches TopK.forward.
  DEAD -- never fires on real traffic (CUDA graph replay bypasses it, see
  that module's docstring and AXIS7.md#2). Kept for the record.
- verify_batch_expert_hooks.py (CAVEMAN_VERIFY_HOOK_OUT): take-3, patches
  ModelRunner.forward, reads the un-finalized TopkCaptureOutput before
  finalize() narrows it to accepted-only tokens. This is the live one --
  see that module's docstring and AXIS7.md#10. Written for sglang 0.5.17.
- install_footprint_pruning.py (CAVEMAN_PRUNING_HOOK_OUT): idea-1 footprint-
  aware draft pruning canary. Patches eagle_worker.select_top_k_tokens.
  Written for sglang 0.4.10 (this repo's A100-compatible environment --
  see that module's docstring for why 0.5.17's sgl-kernel doesn't support
  this GPU architecture). Currently computes but does not yet act on the
  pruned selection -- see that module's docstring.
All three stay cheap to have unconditionally on PYTHONPATH even in
processes that never touch MoE/EAGLE at all, since each install() no-ops
immediately when its own env var is unset.
"""
import os as _os
import sys as _sys
print(f"[caveman-canary] sitecustomize.py loaded, pid={_os.getpid()}, "
      f"CAVEMAN_MOE_HOOK_OUT={_os.environ.get('CAVEMAN_MOE_HOOK_OUT')!r}, "
      f"CAVEMAN_VERIFY_HOOK_OUT={_os.environ.get('CAVEMAN_VERIFY_HOOK_OUT')!r}, "
      f"CAVEMAN_PRUNING_HOOK_OUT={_os.environ.get('CAVEMAN_PRUNING_HOOK_OUT')!r}",
      file=_sys.stderr, flush=True)

from specloop_rt.sglang_patch import moe_expert_hooks as _moe_expert_hooks
from specloop_rt.sglang_patch import verify_batch_expert_hooks as _verify_batch_expert_hooks
from specloop_rt.sglang_patch import install_footprint_pruning as _install_footprint_pruning

_moe_expert_hooks.install()

_num_layers = _os.environ.get("CAVEMAN_VERIFY_HOOK_NUM_LAYERS")
_topk_size = _os.environ.get("CAVEMAN_VERIFY_HOOK_TOPK_SIZE")
if _num_layers and _topk_size:
    _verify_batch_expert_hooks.install(int(_num_layers), int(_topk_size))

_install_footprint_pruning.install()

print(f"[caveman-canary] install() done, pid={_os.getpid()}, "
      f"moe_installed={_moe_expert_hooks._installed}, "
      f"verify_installed={_verify_batch_expert_hooks._installed}, "
      f"pruning_installed={_install_footprint_pruning._installed}",
      file=_sys.stderr, flush=True)
