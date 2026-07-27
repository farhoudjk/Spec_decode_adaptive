"""GPU-free contract test.

Runs WITHOUT vllm, torch, or a GPU. It stubs the vLLM v1 scheduler with a fake
that mimics the documented override surface, then drives the patch through a
synthetic step loop. Purpose: catch signature/attribute drift and verify the
control→actuation→telemetry path BEFORE spending a GPU launch.

    python tests/test_patch_contract.py

What it CANNOT check: that the real vLLM fields (acceptance counts, KV block
attrs) are named as probed — only a real run confirms that (see
vllm_patch/PROVENANCE.md "What to verify on first GPU run"). This test confirms
everything on OUR side of the seam is internally consistent.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------
# Build a fake `vllm` package sufficient for scheduler_patch to import and run.
# --------------------------------------------------------------------------

def _install_fake_vllm():
    vllm = types.ModuleType("vllm")
    v1 = types.ModuleType("vllm.v1")
    core = types.ModuleType("vllm.v1.core")
    sched = types.ModuleType("vllm.v1.core.sched")
    sched_scheduler = types.ModuleType("vllm.v1.core.sched.scheduler")
    sched_output = types.ModuleType("vllm.v1.core.sched.output")

    class _Cfg:
        def __init__(self):
            self.max_num_seqs = 64

    class _SpecCfg:
        def __init__(self):
            self.num_speculative_tokens = 4

    class _VllmConfig:
        def __init__(self):
            self.speculative_config = _SpecCfg()

    class _FreeQ:
        num_free_blocks = 900

    class _BlockPool:
        num_gpu_blocks = 1000
        free_block_queue = _FreeQ()

    class _KVMgr:
        block_pool = _BlockPool()

    class FakeScheduler:
        """Mimics the documented v1 Scheduler override surface."""
        def __init__(self, *a, **k):
            self.scheduler_config = _Cfg()
            self.vllm_config = _VllmConfig()
            self.kv_cache_manager = _KVMgr()
            self.running = []
            self.waiting = []
            self._t = 0

        def schedule(self):
            # emulate admitting up to max_num_seqs, one prefill token each
            want = min(self.scheduler_config.max_num_seqs, 8 + self._t % 5)
            self.running = list(range(want))
            self._t += 1
            rids = [str(i) for i in range(want)]
            out = types.SimpleNamespace(
                num_scheduled_tokens={rid: 1 for rid in rids},
                scheduled_spec_decode_tokens={
                    rid: list(range(self.vllm_config.speculative_config.num_speculative_tokens))
                    for rid in rids},
            )
            return out

        def update_from_output(self, sout, mro):
            return None

    class SchedulerOutput:
        pass

    sched_scheduler.Scheduler = FakeScheduler
    sched_output.SchedulerOutput = SchedulerOutput

    vllm.v1 = v1
    sys.modules["vllm"] = vllm
    sys.modules["vllm.v1"] = v1
    sys.modules["vllm.v1.core"] = core
    sys.modules["vllm.v1.core.sched"] = sched
    sys.modules["vllm.v1.core.sched.scheduler"] = sched_scheduler
    sys.modules["vllm.v1.core.sched.output"] = sched_output
    return FakeScheduler


def _fake_model_output(n_running, k, accept_frac=0.6):
    """Mimics real ModelRunnerOutput's actual v0.9.2 shape: req_ids +
    sampled_token_ids (accepted length is len(tokens)-1, the bonus token is
    always included), NOT the num_accepted_tokens/num_spec_tokens fields the
    patch used to probe for -- those don't exist on the real dataclass (see
    PROVENANCE.md#acceptance)."""
    import random
    req_ids = [str(i) for i in range(n_running)]
    sampled = []
    for _ in range(n_running):
        accepted = max(0, min(k, int(round(k * accept_frac + random.uniform(-1, 1)))))
        sampled.append(list(range(accepted + 1)))  # +1 bonus token, always present
    return types.SimpleNamespace(req_ids=req_ids, sampled_token_ids=sampled)


def run():
    FakeScheduler = _install_fake_vllm()

    from specloop_rt.vllm_patch import SpecLoopScheduler, configure
    from specloop_rt.controllers import build_controller
    from specloop_rt.interface import TelemetryWriter

    tmp = "/tmp/specloop_contract"
    os.makedirs(tmp, exist_ok=True)
    tel = TelemetryWriter(f"{tmp}/steps.jsonl")
    cfg = {"spec": "closed-loop", "admit": "slack", "coordination": "naive",
           "spec_kw": {"gain": 0.5, "period": 1, "gamma_max": 8, "gamma_init": 4},
           "admit_kw": {"gain": 0.5, "period": 1, "init": 64}}
    ctrl = build_controller(cfg, tpot_slo=0.04)
    configure(ctrl, tel, {"gamma_init": 4, "tpot_slo_s": 0.04, "ttft_slo_s": 2.0})

    sch = SpecLoopScheduler()

    gammas, caps, accepts = [], [], []
    for step in range(120):
        out = sch.schedule()
        # feed a fake acceptance back
        mro = _fake_model_output(len(sch.running),
                                 sch.vllm_config.speculative_config.num_speculative_tokens,
                                 accept_frac=0.8 if step < 60 else 0.25)  # step disturbance
        sch.update_from_output(out, mro)
        gammas.append(sch._sl_gamma)
        caps.append(sch.scheduler_config.max_num_seqs)
        accepts.append(sch._sl_ema["accept_rate"].value)
    tel.close()

    # ---- assertions -------------------------------------------------------
    import json
    rows = [json.loads(l) for l in open(f"{tmp}/steps.jsonl") if '"_meta"' not in l]
    assert len(rows) == 120, f"expected 120 telemetry rows, got {len(rows)}"
    assert all("act_gamma" in r for r in rows), "telemetry missing act_gamma"
    assert all(0.0 <= r["accept_rate_ema"] <= 1.0 for r in rows), "accept_rate out of range"
    # gamma must respond to the acceptance drop after step 60
    hi = sum(gammas[30:60]) / 30
    lo = sum(gammas[90:120]) / 30
    assert lo < hi, f"gamma did not fall after acceptance dropped (hi={hi:.2f}, lo={lo:.2f})"
    # KV wiring produced a sane fraction
    assert all(0.0 <= (r["kv_used_blocks"] / max(1, r["kv_total_blocks"])) <= 1.0 for r in rows)
    # actuation actually reached the config object
    assert sch.scheduler_config.max_num_seqs == caps[-1]
    assert sch.vllm_config.speculative_config.num_speculative_tokens == gammas[-1]

    print("PASS contract test")
    print(f"  gamma  hi(step30-60)={hi:.2f}  ->  lo(step90-120)={lo:.2f}  (responded to disturbance)")
    print(f"  accept_rate_ema range: [{min(accepts):.2f}, {max(accepts):.2f}]")
    print(f"  final cap={caps[-1]}  final gamma={gammas[-1]}")
    print(f"  telemetry rows: {len(rows)}")


if __name__ == "__main__":
    run()
