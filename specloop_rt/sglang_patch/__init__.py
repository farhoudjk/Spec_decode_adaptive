"""SGLang-side instrumentation patches, mirroring vllm_patch/'s convention
of documenting monkeypatches in-package rather than forking the engine.

moe_expert_hooks.py is the only module here so far -- see AXIS7.md for why
this exists and PROVENANCE.md for the exact hook point.
"""
