"""GPU-free regression test for the freeze-bug fix (AXIS5_ROOFLINE_MOE.md#4).

Runs WITHOUT vllm, torch, or a GPU. Stubs NgramProposer/EagleProposer/Scheduler
with fakes that mimic the real classes' documented shape (same __init__ /
propose() signatures and the exact frozen-attribute pattern confirmed by
reading vllm==0.9.2 source directly -- see live_gamma_patch.py's module
docstring for citations), then proves:

  1. WITHOUT the patch, mutating speculative_config.num_speculative_tokens
     after construction does NOT change what propose() uses (reproduces the
     bug).
  2. WITH specloop_rt.vllm_patch.live_gamma_patch.apply() called, the SAME
     mutation DOES change what propose() uses, with NO engine/proposer
     rebuild -- this is the actual claim the paper needs to stand behind.
  3. Scheduler.__init__ pins num_lookahead_tokens to the k-ceiling for an
     eagle-mode scheduler (the second, independent freeze point).

    python tests/test_live_gamma_patch.py

What it CANNOT check: that this reaches the real proposer under vLLM's
actual fork/spawn process model for AsyncLLM -- only a real GPU run
confirms that (see replay.py's call site and live_gamma_patch.py's docstring
for the fork-vs-spawn analysis). This test confirms the patch logic itself
is correct in isolation.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_fake_vllm():
    """Fakes mirroring the REAL vllm==0.9.2 classes' frozen-attribute bug,
    confirmed by reading vllm/v1/spec_decode/{ngram_proposer,eagle}.py and
    vllm/v1/core/sched/scheduler.py directly (see live_gamma_patch.py)."""
    vllm = types.ModuleType("vllm")
    v1 = types.ModuleType("vllm.v1")
    spec_decode = types.ModuleType("vllm.v1.spec_decode")
    ngram_mod = types.ModuleType("vllm.v1.spec_decode.ngram_proposer")
    eagle_mod = types.ModuleType("vllm.v1.spec_decode.eagle")
    core = types.ModuleType("vllm.v1.core")
    sched = types.ModuleType("vllm.v1.core.sched")
    sched_scheduler = types.ModuleType("vllm.v1.core.sched.scheduler")

    class FakeSpecCfg:
        def __init__(self, k=4, use_eagle=False):
            self.num_speculative_tokens = k
            self._use_eagle = use_eagle

        def use_eagle(self):
            return self._use_eagle

    class FakeVllmConfig:
        def __init__(self, spec_cfg):
            self.speculative_config = spec_cfg

    class FakeNgramProposer:
        """Mirrors vllm/v1/spec_decode/ngram_proposer.py::NgramProposer:
        self.k copied once in __init__, read fresh inside propose()."""
        def __init__(self, vllm_config):
            self.k = vllm_config.speculative_config.num_speculative_tokens

        def propose(self, context_token_ids):
            return self.k  # stand-in "proposal": just return the k used

    class FakeEagleProposer:
        """Mirrors vllm/v1/spec_decode/eagle.py::EagleProposer: stores
        speculative_config AND a frozen copy self.num_speculative_tokens;
        propose() (pre-patch) uses the frozen copy exclusively."""
        def __init__(self, vllm_config, device=None, runner=None):
            self.vllm_config = vllm_config
            self.speculative_config = vllm_config.speculative_config
            self.num_speculative_tokens = (
                self.speculative_config.num_speculative_tokens)

        def propose(self, *a, **k):
            return self.num_speculative_tokens

    class FakeScheduler:
        """Mirrors vllm/v1/core/sched/scheduler.py::Scheduler's use_eagle /
        num_lookahead_tokens freeze (Scheduler.__init__)."""
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            spec_cfg = vllm_config.speculative_config
            self.use_eagle = bool(spec_cfg and spec_cfg.use_eagle())
            self.num_spec_tokens = self.num_lookahead_tokens = 0
            if spec_cfg:
                self.num_spec_tokens = spec_cfg.num_speculative_tokens
                if self.use_eagle:
                    self.num_lookahead_tokens = self.num_spec_tokens

    ngram_mod.NgramProposer = FakeNgramProposer
    eagle_mod.EagleProposer = FakeEagleProposer
    sched_scheduler.Scheduler = FakeScheduler

    vllm.v1 = v1
    sys.modules["vllm"] = vllm
    sys.modules["vllm.v1"] = v1
    sys.modules["vllm.v1.spec_decode"] = spec_decode
    sys.modules["vllm.v1.spec_decode.ngram_proposer"] = ngram_mod
    sys.modules["vllm.v1.spec_decode.eagle"] = eagle_mod
    sys.modules["vllm.v1.core"] = core
    sys.modules["vllm.v1.core.sched"] = sched
    sys.modules["vllm.v1.core.sched.scheduler"] = sched_scheduler

    return FakeVllmConfig, FakeSpecCfg, FakeNgramProposer, FakeEagleProposer, FakeScheduler


def _fresh_patch_module():
    """live_gamma_patch has a module-level _PATCHED latch; reload it so each
    test function gets a clean, unpatched module (matches a fresh process)."""
    for name in list(sys.modules):
        if name == "specloop_rt.vllm_patch.live_gamma_patch" or name.startswith(
                "specloop_rt.vllm_patch"):
            del sys.modules[name]
    import specloop_rt.vllm_patch.live_gamma_patch as m
    return m


def test_ngram_frozen_without_patch():
    FakeVllmConfig, FakeSpecCfg, FakeNgramProposer, _, _ = _install_fake_vllm()
    _fresh_patch_module()  # imported but NOT applied

    spec_cfg = FakeSpecCfg(k=4)
    vllm_config = FakeVllmConfig(spec_cfg)
    proposer = FakeNgramProposer(vllm_config)
    assert proposer.propose(None) == 4

    spec_cfg.num_speculative_tokens = 8  # live actuation attempt
    assert proposer.propose(None) == 4, (
        "bug reproduction failed: unpatched proposer should stay frozen at "
        "the construction-time k even after the config is mutated")
    print("PASS: unpatched NgramProposer reproduces the freeze bug")


def test_ngram_live_with_patch():
    FakeVllmConfig, FakeSpecCfg, FakeNgramProposer, _, _ = _install_fake_vllm()
    patch = _fresh_patch_module()
    patch.apply()

    spec_cfg = FakeSpecCfg(k=4)
    vllm_config = FakeVllmConfig(spec_cfg)
    proposer = FakeNgramProposer(vllm_config)
    assert proposer.propose(None) == 4

    spec_cfg.num_speculative_tokens = 8
    assert proposer.propose(None) == 8, (
        "patch failed: proposer should now track the live config value "
        "with no rebuild")

    spec_cfg.num_speculative_tokens = 1
    assert proposer.propose(None) == 1, "patch should track decreases too"
    print("PASS: patched NgramProposer tracks live k with no rebuild")


def test_ngram_live_with_patch_real_construction_order():
    """Reproduces vLLM's REAL construction order: GPUModelRunner.__init__
    builds NgramProposer BEFORE EngineCore constructs the Scheduler (which is
    where apply() actually gets called from, via SpecLoopScheduler.__init__ --
    see scheduler_patch.py). This is the order that matters on real hardware;
    test_ngram_live_with_patch above (apply() before construction) is the easy
    case and does NOT exercise the _LIVE_SPEC_CFG fallback this test is for."""
    FakeVllmConfig, FakeSpecCfg, FakeNgramProposer, _, FakeScheduler = _install_fake_vllm()
    patch = _fresh_patch_module()

    spec_cfg = FakeSpecCfg(k=4)
    vllm_config = FakeVllmConfig(spec_cfg)
    proposer = FakeNgramProposer(vllm_config)  # constructed BEFORE apply()
    assert proposer.propose(None) == 4

    patch.apply()  # as if Scheduler.__init__ just ran and called this
    from vllm.v1.core.sched.scheduler import Scheduler
    Scheduler(vllm_config)  # populates _LIVE_SPEC_CFG as a side effect

    spec_cfg.num_speculative_tokens = 8
    assert proposer.propose(None) == 8, (
        "patch failed for a proposer built before apply() ran -- the "
        "_LIVE_SPEC_CFG fallback should have resynced it anyway")
    print("PASS: patched NgramProposer (built before apply()) still tracks "
          "live k via the _LIVE_SPEC_CFG fallback")


def test_eagle_frozen_without_patch():
    FakeVllmConfig, FakeSpecCfg, _, FakeEagleProposer, _ = _install_fake_vllm()
    _fresh_patch_module()

    spec_cfg = FakeSpecCfg(k=4, use_eagle=True)
    vllm_config = FakeVllmConfig(spec_cfg)
    proposer = FakeEagleProposer(vllm_config)
    assert proposer.propose() == 4

    spec_cfg.num_speculative_tokens = 2
    assert proposer.propose() == 4, (
        "bug reproduction failed: unpatched EagleProposer should stay frozen")
    print("PASS: unpatched EagleProposer reproduces the freeze bug")


def test_eagle_live_with_patch():
    FakeVllmConfig, FakeSpecCfg, _, FakeEagleProposer, _ = _install_fake_vllm()
    patch = _fresh_patch_module()
    patch.apply()

    spec_cfg = FakeSpecCfg(k=4, use_eagle=True)
    vllm_config = FakeVllmConfig(spec_cfg)
    proposer = FakeEagleProposer(vllm_config)
    assert proposer.propose() == 4

    spec_cfg.num_speculative_tokens = 7
    assert proposer.propose() == 7, (
        "patch failed: EagleProposer should track the live config value")
    print("PASS: patched EagleProposer tracks live k with no rebuild")


def test_scheduler_lookahead_pinned_for_eagle():
    FakeVllmConfig, FakeSpecCfg, _, _, FakeScheduler = _install_fake_vllm()
    patch = _fresh_patch_module()
    patch.apply()

    # construction-time k=2, well under the 8-token hard ceiling used
    # elsewhere by HillClimbSpec._K_HARD_MAX -- the patched Scheduler should
    # still reserve KV lookahead for k_ceiling=8, not the frozen k=2, so a
    # later live increase to k=8 never under-reserves KV blocks.
    from vllm.v1.core.sched.scheduler import Scheduler
    spec_cfg = FakeSpecCfg(k=2, use_eagle=True)
    vllm_config = FakeVllmConfig(spec_cfg)
    sch = Scheduler(vllm_config)
    assert sch.num_lookahead_tokens == 8, (
        f"expected lookahead pinned to the 8-token ceiling, got "
        f"{sch.num_lookahead_tokens}")
    print("PASS: Scheduler pins EAGLE's num_lookahead_tokens to the k-ceiling")


def test_scheduler_lookahead_untouched_for_ngram():
    FakeVllmConfig, FakeSpecCfg, _, _, FakeScheduler = _install_fake_vllm()
    patch = _fresh_patch_module()
    patch.apply()

    from vllm.v1.core.sched.scheduler import Scheduler
    spec_cfg = FakeSpecCfg(k=4, use_eagle=False)  # ngram: use_eagle() is False
    vllm_config = FakeVllmConfig(spec_cfg)
    sch = Scheduler(vllm_config)
    assert sch.num_lookahead_tokens == 0, (
        "ngram path should be untouched -- it never sets num_lookahead_tokens "
        "in real vLLM, so there is no KV-reservation confound to patch")
    print("PASS: ngram-mode Scheduler's num_lookahead_tokens left at 0 (unaffected)")


def test_scheduler_num_spec_tokens_pinned_for_both_proposer_types():
    """Reproduces the real crash found in scripts/smoketest_live_gamma.py's
    hillclimb run: SpecDecodingStats.new(self.num_spec_tokens) pre-sizes a
    list to the FROZEN construction-time k, and observe_draft() asserts
    every accepted count fits within it. Unlike num_lookahead_tokens this
    applies to BOTH ngram and eagle -- there is no use_eagle gate on
    num_spec_tokens in real vLLM."""
    FakeVllmConfig, FakeSpecCfg, _, _, FakeScheduler = _install_fake_vllm()
    patch = _fresh_patch_module()
    patch.apply()

    from vllm.v1.core.sched.scheduler import Scheduler
    for use_eagle in (False, True):
        spec_cfg = FakeSpecCfg(k=2, use_eagle=use_eagle)
        vllm_config = FakeVllmConfig(spec_cfg)
        sch = Scheduler(vllm_config)
        assert sch.num_spec_tokens == 8, (
            f"expected num_spec_tokens pinned to the 8-token ceiling "
            f"(use_eagle={use_eagle}), got {sch.num_spec_tokens} -- a live "
            f"controller raising k past construction-time k would crash "
            f"SpecDecodingStats.observe_draft()'s assert on this proposer type")
    print("PASS: Scheduler pins num_spec_tokens to the k-ceiling for BOTH "
          "ngram and eagle (the assert this guards is not eagle-gated)")


def test_idempotent():
    _install_fake_vllm()
    patch = _fresh_patch_module()
    patch.apply()
    patch.apply()  # must not double-wrap propose() or raise
    print("PASS: apply() is idempotent")


if __name__ == "__main__":
    test_ngram_frozen_without_patch()
    test_ngram_live_with_patch()
    test_ngram_live_with_patch_real_construction_order()
    test_eagle_frozen_without_patch()
    test_eagle_live_with_patch()
    test_scheduler_lookahead_pinned_for_eagle()
    test_scheduler_lookahead_untouched_for_ngram()
    test_scheduler_num_spec_tokens_pinned_for_both_proposer_types()
    test_idempotent()
    print("\nALL PASS")
