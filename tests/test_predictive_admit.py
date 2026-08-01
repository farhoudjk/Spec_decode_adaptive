"""GPU-free regression tests for the TTFT-predictive admission sensor.

These pin the specific defect the Axis-3 sweeps exposed. On
results_llama3/axis3_llama3_24gb_lowrate/adaptive-cap-only_reason_r2.0_s0, the
ratio-based sensor (TTFTSlackAdmit / _TrendAwareQueueTerm) held the cap at 256
for the entire run: num_waiting peaked at 37 against cap=256, a ratio of 0.14
versus target_util=0.7, so the law saw pure slack -- while the measured TTFT
p99 for that same cell was 37s against a 2.0s SLO, a 19x breach.

The controller was not losing to the static baseline; it never actuated. Test 1
reproduces that exact observation and asserts the two sensors disagree, which
is the whole point of the fix. Tests 2-4 guard the ways a "tighten harder"
change can be wrong: it must not tighten an idle system, must still respect the
KV ceiling, and must release the cap once the queue drains.

    python tests/test_predictive_admit.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specloop_rt.controllers import (KVPredictiveAdmit, TTFTPredictiveAdmit,
                                     TTFTSlackAdmit)
from specloop_rt.interface import StepObservation

EST_LEN = 840.0   # measured mean output length for reason/CNN-DailyMail


def _obs(step, waiting, running, tpot=0.040, kv=0.5, cap=256):
    return StepObservation(
        t_wall=0.0, step=step, num_running=running, num_waiting=waiting,
        num_scheduled_tokens=0, num_spec_tokens=0,
        kv_used_blocks=int(kv * 1000), kv_total_blocks=1000,
        accepted_tokens=0, proposed_tokens=0, accept_rate_ema=0.5,
        accepted_per_req_ema=1.0, step_time_ema=0.04, tpot_ema=tpot,
        gamma_current=1, max_num_seqs_current=cap,
        tpot_slo_s=0.040, ttft_slo_s=2.0)


def _settle(ctrl, ob_fn, steps=40):
    act = None
    for step in range(0, steps + 1, 4):
        act = ctrl.on_step(ob_fn(step))
    return act.max_num_seqs


def test_predictive_fires_where_ratio_is_numb():
    """The rate=2 observation that left the old sensor inert."""
    state = lambda s: _obs(s, waiting=37, running=22, tpot=0.039, cap=256)

    old_cap = _settle(TTFTSlackAdmit(batch_max=256, init=256), state)
    new = TTFTPredictiveAdmit(batch_max=256, init=256, est_output_len=EST_LEN)
    new_cap = _settle(new, state)

    assert old_cap == 256, f"ratio sensor unexpectedly moved: {old_cap}"
    assert new_cap < 64, f"predictive sensor failed to tighten: {new_cap}"

    predicted = new._queue_term.predicted_wait(state(0))
    assert predicted > 2.0, f"predicted wait {predicted:.1f}s should exceed the 2.0s SLO"
    print(f"  ratio sensor cap={old_cap} (numb) | predictive cap={new_cap}, "
          f"predicted wait={predicted:.1f}s vs SLO 2.0s")


def test_idle_system_is_not_throttled():
    """A tighten-only law would be useless; an empty queue must let the cap grow."""
    cap = _settle(TTFTPredictiveAdmit(batch_max=256, init=64, est_output_len=EST_LEN),
                  lambda s: _obs(s, waiting=0, running=2, tpot=0.020))
    assert cap > 64, f"cap should grow when idle, got {cap}"
    print(f"  idle: cap 64 -> {cap}")


def test_kv_ceiling_still_binds():
    """KV safety bound must survive the sensor swap: min(ttft_term, kv_ceiling)."""
    cap = _settle(KVPredictiveAdmit(batch_max=256, init=256, est_output_len=EST_LEN),
                  lambda s: _obs(s, waiting=0, running=30, tpot=0.020, kv=0.95))
    over = (0.95 - 0.80) / 0.20
    expected = 256 - over * (256 - 0.30 * 256)
    assert abs(cap - expected) <= 2, f"cap {cap} != KV ceiling {expected:.0f}"
    print(f"  kv=0.95: cap={cap} (ceiling {expected:.0f})")


def test_cap_recovers_when_queue_drains():
    """Tighten then release: the loop must not latch at the floor."""
    ctrl = TTFTPredictiveAdmit(batch_max=256, init=256, est_output_len=EST_LEN)
    tight = _settle(ctrl, lambda s: _obs(s, waiting=37, running=22, tpot=0.039))
    # queue drains; same controller instance continues
    act = None
    for step in range(44, 200, 4):
        act = ctrl.on_step(_obs(step, waiting=0, running=4, tpot=0.020))
    assert act.max_num_seqs > tight, f"cap latched at {tight} -> {act.max_num_seqs}"
    print(f"  drain: cap {tight} -> {act.max_num_seqs}")


if __name__ == "__main__":
    for fn in [test_predictive_fires_where_ratio_is_numb,
               test_idle_system_is_not_throttled,
               test_kv_ceiling_still_binds,
               test_cap_recovers_when_queue_drains]:
        print(f"{fn.__name__}:")
        fn()
    print("\nAll predictive-admission contract tests passed.")
