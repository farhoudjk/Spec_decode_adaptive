"""Real-GPU smoke test for the freeze-bug fix (AXIS5_ROOFLINE_MOE.md#4).

Runs ONE engine (no rebuild), with a controller that flips gamma from 4 to 1
at a fixed step, and checks the GROUND-TRUTH signal -- StepObservation.
num_spec_tokens, which scheduler_patch.py's _sl_sum_spec() reads directly off
the real SchedulerOutput.scheduled_spec_decode_tokens, i.e. what the real
proposer actually emitted -- not gamma_current, which is only what the
controller believes it set (that distinction IS the bug: before the patch,
gamma_current moves and num_spec_tokens does not).

Usage:
    python3 scripts/smoketest_live_gamma.py --config configs/smoketest_live_gamma.yaml

Passes iff mean per-request proposed spec-tokens/step during the k=1 phase is
meaningfully lower than during the k=4 phase, sampled from the SAME engine
instance (no subprocess relaunch, no fresh proposer).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specloop_rt import workload as W
from specloop_rt.interface import Controller, ControlAction


class StepGammaSpec(Controller):
    """Test-only controller: gamma_a for the first `flip_step` scheduler
    steps, then gamma_b forever after. No acceptance/ITL sensing -- this
    isolates "does live actuation reach the proposer" from any control-law
    question."""
    name = "step-gamma-spec"

    def __init__(self, gamma_a: int = 4, gamma_b: int = 1, flip_step: int = 60):
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b
        self.flip_step = flip_step

    def on_step(self, obs):
        g = self.gamma_a if obs.step < self.flip_step else self.gamma_b
        return ControlAction(gamma=g)


class StaticAdmit(Controller):
    name = "static-admit-test"

    def __init__(self, max_num_seqs: int = 16):
        self.cap = max_num_seqs

    def on_step(self, obs):
        return ControlAction(max_num_seqs=self.cap)


class TestComposite(Controller):
    name = "test-composite"

    def __init__(self, spec, admit):
        self.spec = spec
        self.admit = admit

    def on_step(self, obs):
        a = self.spec.on_step(obs)
        b = self.admit.on_step(obs)
        return ControlAction(gamma=a.gamma, max_num_seqs=b.max_num_seqs)


async def main_async(cfg, out_dir, flip_step, gamma_a, gamma_b, rate, duration):
    # NOTE: live_gamma_patch is NOT applied here in the client process -- see
    # the comment in replay.py::_build_engine for why doing so would force
    # vLLM's EngineCore child to `spawn` instead of `fork`, which breaks
    # configure()'s process-global controller/telemetry propagation. It's
    # applied inside SpecLoopScheduler.__init__ instead, which always runs in
    # the right process.
    from specloop_rt.vllm_patch import configure
    from specloop_rt.interface import TelemetryWriter
    from specloop_rt.replay import _build_engine, _run_one

    os.makedirs(out_dir, exist_ok=True)
    controller = TestComposite(StepGammaSpec(gamma_a, gamma_b, flip_step),
                               StaticAdmit(cfg["runtime"]["max_num_seqs_init"]))
    telemetry = TelemetryWriter(os.path.join(out_dir, "steps.jsonl"))
    configure(controller, telemetry, cfg["runtime"])

    engine = _build_engine(cfg)

    from vllm import SamplingParams
    def sampling(mt):
        return SamplingParams(temperature=0.0, max_tokens=mt, ignore_eos=True)

    trace = W.homogeneous(rtype="code", rate=rate, seed=0, duration=duration,
                          use_real_corpus=False)

    import time
    results = {}
    t0 = time.monotonic()
    tasks = [asyncio.create_task(_run_one(engine, r, sampling, t0, results))
            for r in trace]
    await asyncio.gather(*tasks)
    telemetry.close()
    return os.path.join(out_dir, "steps.jsonl")


def analyze(steps_path, flip_step):
    rows = [json.loads(l) for l in open(steps_path) if '"_meta"' not in l]
    if not rows:
        print("FAIL: no telemetry rows recorded")
        return False

    pre = [r for r in rows if r["step"] < flip_step and r["num_running"] > 0]
    post = [r for r in rows if r["step"] >= flip_step and r["num_running"] > 0]
    if not pre or not post:
        print(f"FAIL: not enough steps on one side of the flip "
              f"(pre={len(pre)}, post={len(post)}) -- rerun with longer --duration")
        return False

    def per_req_spec(rows):
        vals = [r["num_spec_tokens"] / r["num_running"] for r in rows]
        return sum(vals) / len(vals)

    pre_spec = per_req_spec(pre)
    post_spec = per_req_spec(post)
    pre_gamma = sum(r["gamma_current"] for r in pre) / len(pre)
    post_gamma = sum(r["gamma_current"] for r in post) / len(post)

    print(f"gamma_current:        pre={pre_gamma:.2f}  post={post_gamma:.2f}  "
          f"(what the controller THINKS it set)")
    print(f"num_spec_tokens/req:  pre={pre_spec:.3f}  post={post_spec:.3f}  "
          f"(what the REAL proposer actually emitted -- ground truth)")

    if pre_spec <= post_spec * 1.2:
        print("\nFAIL: num_spec_tokens/req did not drop after the gamma flip --"
              " this is the freeze bug (or the patch did not apply).")
        return False
    print("\nPASS: num_spec_tokens/req tracks the live gamma flip -- the real "
          "proposer is responding to live actuation without an engine rebuild.")
    return True


def main(argv=None):
    p = argparse.ArgumentParser("live_gamma_patch smoke test")
    p.add_argument("--config", required=True)
    p.add_argument("--gamma-a", type=int, default=4)
    p.add_argument("--gamma-b", type=int, default=1)
    p.add_argument("--flip-step", type=int, default=60)
    p.add_argument("--rate", type=float, default=8.0)
    p.add_argument("--duration", type=float, default=45.0)
    p.add_argument("--out", default="results_gpu_sweep/smoketest_live_gamma")
    a = p.parse_args(argv)

    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    steps_path = asyncio.run(main_async(cfg, a.out, a.flip_step, a.gamma_a,
                                        a.gamma_b, a.rate, a.duration))
    ok = analyze(steps_path, a.flip_step)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
