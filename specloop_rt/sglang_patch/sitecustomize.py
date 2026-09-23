import os as _os
import sys as _sys

from specloop_rt.sglang_patch import verify_batch_expert_hooks as _verify_batch_expert_hooks
from specloop_rt.sglang_patch import ecospec_collect as _ecospec_collect

_num_layers = _os.environ.get("CAVEMAN_VERIFY_HOOK_NUM_LAYERS")
_topk_size = _os.environ.get("CAVEMAN_VERIFY_HOOK_TOPK_SIZE")
if _num_layers and _topk_size:
    _verify_batch_expert_hooks.install(int(_num_layers), int(_topk_size))

_ecospec_collect.install()
