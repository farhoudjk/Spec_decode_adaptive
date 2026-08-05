"""Patch that makes vLLM 0.9.2's speculation proposers re-read k every step,
closing the freeze bug documented in AXIS5_ROOFLINE_MOE.md#4.

  ROOT CAUSE (confirmed by reading vllm==0.9.2 source directly, both classes):

    vllm/v1/spec_decode/ngram_proposer.py, NgramProposer.__init__:
        self.k = vllm_config.speculative_config.num_speculative_tokens
    vllm/v1/spec_decode/eagle.py, EagleProposer.__init__:
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

  Both copy the config value into an instance attribute ONCE and their
  propose() methods read the frozen copy forever after. scheduler_patch.py's
  _sl_apply_gamma() writes a new value into
  vllm_config.speculative_config.num_speculative_tokens every control step --
  that write is real (SpeculativeConfig is a plain, non-frozen dataclass with
  no post-construction validation reruns) but nothing downstream ever looks
  at the config object again, so it is silently discarded.

  This module monkeypatches propose() on both proposer classes (via
  __init__ wrapping, not vllm source edits) to resync the instance attribute
  from the live speculative_config at the top of every propose() call. This
  is safe because neither proposer pre-allocates any k-sized buffer in
  __init__ -- verified by reading both files: EagleProposer's persistent
  cudagraph buffers are sized by max_num_tokens/max_num_seqs, never by
  num_speculative_tokens; its only two USES of the cached value are the
  early-exit check and the `for _ in range(self.num_speculative_tokens - 1)`
  generation loop, both re-evaluated fresh on every propose() call once the
  attribute is resynced. NgramProposer's self.k is read fresh inside
  propose() already (`k = min(self.k, ...)`), so resyncing self.k before that
  line is sufficient.

  SECOND, INDEPENDENT FREEZE POINT (scheduler-side, EAGLE only):

    vllm/v1/core/sched/scheduler.py, Scheduler.__init__:
        self.num_lookahead_tokens = self.num_spec_tokens   # only if use_eagle

  num_lookahead_tokens is passed to KVCacheManager.allocate_slots() every
  step to reserve KV blocks for the eagle proposer's own draft-model KV
  cache. It is set once at construction from the same frozen k and never
  resynced. Raising live k WITHOUT fixing this risks under-reserving KV
  blocks for EAGLE once actuated k exceeds construction-time k, which
  degrades to spurious preemption (allocate_slots returns None, not memory
  corruption -- confirmed by reading allocate_slots: it just returns None
  when free blocks are insufficient) rather than a crash, but still an
  unwanted confound. Fixed here by pinning num_lookahead_tokens to the
  controller's hard k-ceiling once at construction (over-provision instead
  of resync-per-step, simpler and avoids a live scheduler-object mutation
  in this already-fragile seam). ngram does NOT use num_lookahead_tokens at
  all (only EAGLE sets it nonzero, see the `if speculative_config.use_eagle()`
  gate) -- there is no KV-reservation confound for the ngram path this repo's
  Mixtral roofline work actually uses.

  THIRD, INDEPENDENT FREEZE POINT (scheduler-side, BOTH proposer types --
  found the hard way, via a live crash, not by reading source first):

    vllm/v1/core/sched/scheduler.py, Scheduler.__init__:
        self.num_spec_tokens = speculative_config.num_speculative_tokens
    vllm/v1/core/sched/scheduler.py, Scheduler.update_from_output():
        spec_decoding_stats = self.make_spec_decoding_stats(...)
          -> SpecDecodingStats.new(self.num_spec_tokens)
             # pre-sizes num_accepted_tokens_per_pos = [0] * num_spec_tokens
    vllm/v1/spec_decode/metrics.py, SpecDecodingStats.observe_draft():
        assert num_accepted_tokens <= self.num_spec_tokens

  self.num_spec_tokens is frozen at construction exactly like
  num_lookahead_tokens, but unlike num_lookahead_tokens it is NOT eagle-only
  -- every proposer type builds SpecDecodingStats from it every step. Raising
  live k above the construction-time value (exactly what HillClimbSpec does
  the first time it climbs past gamma_init) lets a request accept more
  tokens than num_spec_tokens allows, which trips the assert and takes down
  the whole EngineCore process (AssertionError -> EngineDeadError propagated
  to the client). This was NOT found by reading the source first -- it
  surfaced as a real crash the first time a live controller was allowed to
  raise k past its start value on this patch, in scripts/
  smoketest_live_gamma.py's hillclimb run. Fixed the same way as
  num_lookahead_tokens: pin self.num_spec_tokens to the k-ceiling at
  construction too, for every proposer type (not just eagle).

  Applies only under vLLM's single-process executors (UniProcExecutor /
  ExecutorWithExternalLauncher) -- i.e. no tensor parallelism -- which is
  what every config in this repo uses (1x A100 / 1x A5000, TP=1). Under
  multi-GPU TP, the proposer instance lives in a separate worker process and
  this in-process monkeypatch would not reach it; that case is out of scope
  (see PROVENANCE.md's own scope note) and this module does not attempt it.

  CALL SITE: apply() is called from SpecLoopScheduler.__init__
  (scheduler_patch.py), NOT from replay.py's client process. That is not
  just belt-and-suspenders -- it is load-bearing. Importing EagleProposer
  (needed to patch it) initializes a CUDA context as a side effect of its
  own import chain; doing that in the CLIENT process before engine
  construction flips vLLM's EngineCore child from `fork` to `spawn`
  (vllm.utils.get_mp_context forces spawn once CUDA is initialized in the
  parent). Under fork the child inherits the parent's memory, including
  scheduler_patch.py's `configure()`-installed globals; under spawn it gets
  a fresh interpreter and those globals silently reset to their unconfigured
  defaults (_CONTROLLER=None, _TELEMETRY=None) -- confirmed directly: a
  real replay.py run that imported this module in the client process wrote
  ONLY the _meta header to steps.jsonl (zero step rows), because the
  scheduler's _TELEMETRY was None the entire run. Calling apply() only from
  inside Scheduler.__init__ (which always executes in the EngineCore
  process, whichever start method vLLM chose) avoids the client-process
  import entirely.
"""
from __future__ import annotations

import functools

_PATCHED = False

# Process-global fallback handle to the live SpeculativeConfig, set by
# _patch_scheduler_lookahead()'s Scheduler.__init__ wrapper. NgramProposer is
# constructed by GPUModelRunner.__init__, which vllm's EngineCore.__init__
# runs BEFORE constructing the Scheduler (confirmed by reading
# vllm/v1/engine/core.py: self.model_executor = executor_class(vllm_config)
# precedes self.scheduler = Scheduler(...)) -- so by the time apply() runs
# (from inside Scheduler.__init__, see scheduler_patch.py), any ngram
# proposer instance already exists and never received the __init__-time
# _sl_spec_cfg stash this patch would otherwise rely on. propose() falls back
# to this global so an already-constructed instance still gets resynced.
_LIVE_SPEC_CFG = None


def apply() -> None:
    """Monkeypatch NgramProposer/EagleProposer to re-read k every propose()
    call, and pin EAGLE's KV lookahead reservation to a safe ceiling.

    Idempotent -- safe to call more than once (e.g. once per sweep-script
    subprocess); the second call is a no-op. Also a no-op (with a printed
    warning, not a raise) if vllm.v1.spec_decode isn't importable -- this
    happens under tests/test_patch_contract.py's stub vllm tree, which fakes
    just enough of vllm.v1.core.sched.scheduler to exercise the control/
    telemetry path without a real vLLM install, and correctly has no
    vllm.v1.spec_decode at all. A REAL vLLM install missing this module
    would be a genuine version mismatch worth knowing about, hence the
    warning rather than a silent pass.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from vllm.v1.spec_decode.ngram_proposer import NgramProposer
        from vllm.v1.spec_decode.eagle import EagleProposer
    except ModuleNotFoundError:
        print("live_gamma_patch.apply(): vllm.v1.spec_decode not importable "
              "(stub environment, or a vLLM version this patch isn't pinned "
              "to) -- skipping the freeze-bug fix. Real GPU runs must see "
              "vllm==0.9.2 installed for this NOT to print.")
        return

    _patch_ngram(NgramProposer)
    _patch_eagle(EagleProposer)
    _patch_scheduler_lookahead()

    _PATCHED = True


def _patch_ngram(cls) -> None:
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, vllm_config, *a, **kw):
        orig_init(self, vllm_config, *a, **kw)
        # Stash a live handle for instances constructed AFTER apply() (e.g. a
        # future vLLM version that reorders construction) -- see propose()'s
        # fallback to the module global for the (current, confirmed) case
        # where construction happens first.
        self._sl_spec_cfg = vllm_config.speculative_config

    orig_propose = cls.propose

    @functools.wraps(orig_propose)
    def propose(self, context_token_ids):
        spec_cfg = getattr(self, "_sl_spec_cfg", None) or _LIVE_SPEC_CFG
        if spec_cfg is not None:
            self.k = spec_cfg.num_speculative_tokens
        return orig_propose(self, context_token_ids)

    cls.__init__ = __init__
    cls.propose = propose


def _patch_eagle(cls) -> None:
    # EagleProposer already stores self.speculative_config in __init__
    # (vllm/v1/spec_decode/eagle.py) -- no __init__ patch needed, only
    # propose() needs to resync before using the cached count.
    orig_propose = cls.propose

    @functools.wraps(orig_propose)
    def propose(self, *args, **kwargs):
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        return orig_propose(self, *args, **kwargs)

    cls.propose = propose


def _patch_scheduler_lookahead(k_ceiling: int = 8) -> None:
    """Pin two scheduler-side freeze points to k_ceiling at construction, so
    a live k increase (within [0, k_ceiling]) never trips a stale-k
    assumption baked in at Scheduler construction time:

      * num_lookahead_tokens (EAGLE only) -- KV block reservation for the
        eagle proposer's own draft-model KV cache.
      * num_spec_tokens (ALL proposer types) -- pre-sizes
        SpecDecodingStats.num_accepted_tokens_per_pos and gates an assert in
        observe_draft(); this one is NOT eagle-gated, see the "THIRD freeze
        point" section of this module's docstring for how it was found (a
        live crash, not a source read).

    k_ceiling must match (or exceed) the hard max any live controller can
    actuate to -- see HillClimbSpec._K_HARD_MAX in controllers.py.

    Also publishes the live SpeculativeConfig to the module-level
    _LIVE_SPEC_CFG fallback (see its definition above) -- Scheduler.__init__
    is the earliest point in EngineCore construction where apply() actually
    runs (see scheduler_patch.py), and it always has vllm_config on self, so
    this is the one reliable place to capture it for NgramProposer.propose()
    to fall back to when its own instance was built earlier.
    """
    from vllm.v1.core.sched import scheduler as sched_mod

    orig_init = sched_mod.Scheduler.__init__

    @functools.wraps(orig_init)
    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        global _LIVE_SPEC_CFG
        _LIVE_SPEC_CFG = getattr(self.vllm_config, "speculative_config", None)
        if _LIVE_SPEC_CFG is not None:
            self.num_spec_tokens = max(self.num_spec_tokens, k_ceiling)
        if self.use_eagle:
            self.num_lookahead_tokens = max(self.num_lookahead_tokens,
                                            k_ceiling)

    sched_mod.Scheduler.__init__ = __init__
