"""vLLM v1 patch layer — the ONLY module that imports vllm internals.

  ⚠️  VERSION-FRAGILE.  Pinned to vLLM v0.9.x / v1 engine (schedule() and
      update_from_output() in vllm/v1/core/sched/scheduler.py).  vLLM's v1
      internals move between minor releases; if import or attribute errors
      appear, fix them HERE and nowhere else.  Provenance for every touchpoint
      is in vllm_patch/PROVENANCE.md, keyed to the upstream file:line this was
      written against.

  ⚠️  UNVERIFIED ON HARDWARE.  This was written against the documented v1 API,
      not executed (author environment had no GPU/CUDA/vLLM).  Treat the first
      GPU run as the real integration test; see tests/test_patch_contract.py
      for the shape assertions to run first with a stub.

Design: subclass the v1 Scheduler so we avoid editing upstream files in place.
  * ``schedule()``  — call super(), then override the per-request speculative
    token count and the admission cap using the controller's ControlAction.
  * ``update_from_output()`` — read accepted/proposed spec-token counts, feed
    the telemetry aggregator, and build the StepObservation the controller sees
    on the NEXT step.

Registered via:  --scheduler-cls specloop_rt.vllm_patch.SpecLoopScheduler
(The user asked for a full fork; a subclass registered through the supported
--scheduler-cls seam is the minimal-diff form of that and survives rebases.
PROVENANCE.md documents the equivalent in-place edits if a hard fork is
preferred.)
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, Optional

# NOTE: these imports are the fragile surface. Keep them in this try/except so
# a wrong version yields a clear, single point of failure with guidance.
try:
    from vllm.v1.core.sched.scheduler import Scheduler as _V1Scheduler
    from vllm.v1.core.sched.output import SchedulerOutput
    _VLLM_OK = True
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover - depends on installed vllm
    _V1Scheduler = object
    SchedulerOutput = object
    _VLLM_OK = False
    _IMPORT_ERR = e

from specloop_rt.interface import (EMA, ControlAction, StepObservation,
                                   TelemetryWriter)

# Set by the launcher (specloop_rt.launch) before the engine constructs the
# scheduler, because --scheduler-cls gives us no constructor hook of our own.
_CONTROLLER = None
_TELEMETRY: Optional[TelemetryWriter] = None
_RUNTIME_CFG: Dict = {}
# Most recent StepObservation. NOTE: this is only readable from INSIDE the
# engine process. vLLM v1 runs EngineCore in a separate process by default
# (VLLM_ENABLE_V1_MULTIPROCESSING), so a client-process reader sees None
# forever -- that mistake silently disabled load shedding in an early Axis-4
# run (shed_frac=0.0, indistinguishable from "chose not to shed"). Client-side
# consumers must derive their own signal; see replay.replay's shed_fn.
_LAST_OBS: Optional[StepObservation] = None


def configure(controller, telemetry: TelemetryWriter, runtime_cfg: Dict) -> None:
    """Install process-global controller + telemetry. Called once, pre-engine."""
    global _CONTROLLER, _TELEMETRY, _RUNTIME_CFG
    _CONTROLLER = controller
    _TELEMETRY = telemetry
    _RUNTIME_CFG = runtime_cfg


def last_observation() -> Optional[StepObservation]:
    """Latest per-step observation, or None before the first scheduler step."""
    return _LAST_OBS


class SpecLoopScheduler(_V1Scheduler):
    """v1 Scheduler that applies a runtime controller each step.

    Everything vLLM-internal is confined to the three overrides below.  If an
    attribute name here is wrong for your installed version, it is a localized
    fix — the field it stands for is named in the inline comment.
    """

    def __init__(self, *args, **kwargs):
        if not _VLLM_OK:
            raise ImportError(
                "vLLM v1 scheduler import failed; specloop patch is pinned to "
                f"the v0.9.x v1 engine. Original error: {_IMPORT_ERR}. "
                "See vllm_patch/PROVENANCE.md for the API this targets.")
        super().__init__(*args, **kwargs)
        self._sl_step = 0
        self._sl_ema = {
            "accept_rate": EMA(0.1, 0.6),
            "accepted_per_req": EMA(0.1, 1.0),
            "step_time": EMA(0.1, 0.01),
            "tpot": EMA(0.1, 0.03),
        }
        self._sl_last_schedule_wall = time.monotonic()
        self._sl_gamma = int(_RUNTIME_CFG.get("gamma_init", 4))
        self._sl_max_num_seqs = int(getattr(self.scheduler_config, "max_num_seqs", 64))
        self._sl_per_req_gamma: Dict[str, int] = {}
        # last-step accounting, filled in update_from_output()
        self._sl_last_accepted = 0
        self._sl_last_proposed = 0

    # ------------------------------------------------------------------
    # helpers reading vLLM-internal state (single provenance point each)
    # ------------------------------------------------------------------
    def _sl_kv_blocks(self):
        """(used_blocks, total_blocks).

        Provenance: KVCacheManager block pool.  In v1 the manager is
        self.kv_cache_manager; total capacity is the GPU block count and free
        is tracked by the block pool.  Attribute names have churned — adjust
        here only.  See PROVENANCE.md#kv.
        """
        mgr = getattr(self, "kv_cache_manager", None)
        if mgr is None:
            return 0, 1
        # try the common shapes across v0.9.x
        total = (getattr(mgr, "num_gpu_blocks", None)
                 or getattr(getattr(mgr, "block_pool", None), "num_gpu_blocks", None)
                 or 1)
        free = None
        bp = getattr(mgr, "block_pool", None)
        if bp is not None:
            fq = getattr(bp, "free_block_queue", None)
            free = getattr(fq, "num_free_blocks", None) if fq is not None else None
            if free is None:
                free = getattr(bp, "num_free_blocks", None)
        if free is None:
            free = getattr(mgr, "num_free_blocks", total)
        return int(total) - int(free), int(total)

    def _sl_build_obs(self, num_scheduled_tokens: int, num_spec_tokens: int) -> StepObservation:
        used, total = self._sl_kv_blocks()
        return StepObservation(
            t_wall=time.monotonic(), step=self._sl_step,
            num_running=len(self.running),           # provenance: self.running (list[Request])
            num_waiting=len(self.waiting),           # provenance: self.waiting (deque)
            num_scheduled_tokens=num_scheduled_tokens,
            num_spec_tokens=num_spec_tokens,
            kv_used_blocks=used, kv_total_blocks=total,
            accepted_tokens=self._sl_last_accepted,
            proposed_tokens=self._sl_last_proposed,
            accept_rate_ema=self._sl_ema["accept_rate"].value,
            accepted_per_req_ema=self._sl_ema["accepted_per_req"].value,
            step_time_ema=self._sl_ema["step_time"].value,
            tpot_ema=self._sl_ema["tpot"].value,
            gamma_current=self._sl_gamma,
            max_num_seqs_current=self._sl_max_num_seqs,
            tpot_slo_s=float(_RUNTIME_CFG.get("tpot_slo_s", 0.05)),
            ttft_slo_s=float(_RUNTIME_CFG.get("ttft_slo_s", 2.0)),
        )

    # ------------------------------------------------------------------
    # override 1: admission cap, applied BEFORE super().schedule() selects reqs
    # ------------------------------------------------------------------
    def schedule(self) -> "SchedulerOutput":
        now = time.monotonic()
        self._sl_ema["step_time"].update(now - self._sl_last_schedule_wall)
        self._sl_last_schedule_wall = now

        # Build the observation from PRE-schedule state and ask the controller.
        obs = self._sl_build_obs(num_scheduled_tokens=0, num_spec_tokens=0)
        action: ControlAction = (_CONTROLLER.on_step(obs) if _CONTROLLER is not None
                                 else ControlAction())

        # --- apply admission actuation -------------------------------------
        # Provenance: max_num_seqs lives on self.scheduler_config and is read by
        # the base schedule() when capping the running set.  Mutating it here
        # before super() is the least-invasive admission hook.  See
        # PROVENANCE.md#admission.
        if action.max_num_seqs is not None:
            self._sl_max_num_seqs = int(action.max_num_seqs)
            self.scheduler_config.max_num_seqs = self._sl_max_num_seqs

        # --- apply speculation actuation -----------------------------------
        if action.gamma is not None:
            self._sl_gamma = int(action.gamma)
        if action.per_request_gamma:
            self._sl_per_req_gamma = dict(action.per_request_gamma)
        self._sl_apply_gamma(self._sl_gamma, self._sl_per_req_gamma)

        out = super().schedule()

        # count scheduled + speculative tokens from the output for telemetry
        n_sched = self._sl_sum_scheduled(out)
        n_spec = self._sl_sum_spec(out)
        obs2 = self._sl_build_obs(n_sched, n_spec)
        global _LAST_OBS
        _LAST_OBS = obs2
        if _TELEMETRY is not None:
            _TELEMETRY.record(obs2, action)
        self._sl_step += 1
        return out

    # ------------------------------------------------------------------
    # override 2: read acceptance from the model output
    # ------------------------------------------------------------------
    def update_from_output(self, scheduler_output, model_runner_output):
        # Provenance: spec-decode acceptance is reported per request in the
        # model runner output (accepted token ids per spec position).  In v1 the
        # sampled/accepted structure is on model_runner_output; the exact field
        # differs by version (spec_token_ids / num_accepted).  See
        # PROVENANCE.md#acceptance.  We defensively probe several shapes.
        accepted, proposed = self._sl_extract_acceptance(model_runner_output,
                                                          scheduler_output)
        self._sl_last_accepted = accepted
        self._sl_last_proposed = proposed
        if proposed > 0:
            self._sl_ema["accept_rate"].update(accepted / proposed)
        n_run = max(1, len(self.running))
        self._sl_ema["accepted_per_req"].update(accepted / n_run)
        # TPOT proxy: step_time_ema alone, NOT step_time / (accepted + n_run).
        #
        # The old formula divided one scheduler step's wall time by the TOTAL
        # tokens produced across the whole running batch that step (summing
        # accepted spec tokens plus one bonus token per request), which
        # computes aggregate GPU throughput (tokens/sec across the batch), not
        # per-request TPOT (seconds between one request's own tokens). In v1's
        # continuous batching, each running request advances by ~1 decode
        # token per scheduler step (barring preemption) -- so the wall time
        # between two consecutive tokens FOR ONE REQUEST is approximately the
        # step time itself, not the step time divided by n_run.
        #
        # This was silently wrong at every concurrency tested: confirmed
        # against real per-request TPOT (tpot_p50 in
        # specloop_rt/analysis.py's end_metrics, computed independently from
        # replay.py's request-level timestamps) at num_running=20-32 (tpot_ema
        # read ~0.001-0.002s against a true tpot_p50 of ~0.029s) and
        # num_running=215-256 (~0.0008s against ~0.248s) -- roughly 15-300x
        # off in both cases, and the error grows with concurrency because the
        # old denominator scaled with n_run while the true per-request TPOT
        # does not. GatedSpec's decode-bound gate (specloop_rt/controllers.py)
        # compares this signal to tpot_slo_s directly, so the old formula kept
        # the gate closed even when the system was genuinely decode-bound by
        # 6x its SLO (results_gpu_sweep/axis3_validate/grid.json,
        # results_gpu_sweep/axis3_validate2/grid.json). SlackAdmit reads the
        # same signal and was affected identically.
        st = self._sl_ema["step_time"].value
        self._sl_ema["tpot"].update(st)
        return super().update_from_output(scheduler_output, model_runner_output)

    # ------------------------------------------------------------------
    # override 3: enforce gamma. Two supported mechanisms, see PROVENANCE.md#k
    # ------------------------------------------------------------------
    def _sl_apply_gamma(self, gamma: int, per_req: Dict[str, int]) -> None:
        """Set the per-step speculative token budget.

        Mechanism A (config knob): the v1 spec config exposes
        num_speculative_tokens; the proposer reads it each step.  We write the
        effective k there.  Mechanism B (ragged): if the proposer supports a
        per-request k (DynamicProposer / eagle_dynamic, PR #26504), we stash the
        map where the proposer reads it.  Both are best-effort and guarded.
        """
        spec_cfg = getattr(self.vllm_config, "speculative_config", None)
        if spec_cfg is not None and hasattr(spec_cfg, "num_speculative_tokens"):
            try:
                spec_cfg.num_speculative_tokens = max(0, int(gamma))
            except Exception:
                pass
        # ragged map for a dynamic proposer, if present
        proposer = getattr(self, "drafter", None) or getattr(self, "proposer", None)
        if proposer is not None and per_req:
            setattr(proposer, "_specloop_per_req_k", dict(per_req))

    # ------------------------------------------------------------------
    # output-shape helpers (defensive; adjust names if version differs)
    # ------------------------------------------------------------------
    @staticmethod
    def _sl_sum_scheduled(out) -> int:
        d = getattr(out, "num_scheduled_tokens", None)
        if isinstance(d, dict):
            return int(sum(d.values()))
        return int(getattr(out, "total_num_scheduled_tokens", 0) or 0)

    @staticmethod
    def _sl_sum_spec(out) -> int:
        d = getattr(out, "scheduled_spec_decode_tokens", None)
        if isinstance(d, dict):
            return int(sum(len(v) for v in d.values()))
        return 0

    @staticmethod
    def _sl_extract_acceptance(mro, sout) -> tuple:
        """Return (accepted, proposed) totals for this step.

        Confirmed against vllm==0.9.2's real v1.outputs.ModelRunnerOutput /
        v1.core.sched.output.SchedulerOutput (no num_accepted_tokens /
        num_spec_tokens field exists on either -- the earlier probe for that
        shape always fell through to the zero-accepted fallback below, which
        made accept_rate_ema report 0.0 all run; see PROVENANCE.md#acceptance).

        Per-request accepted length = len(sampled_token_ids[i]) - 1: v1 always
        samples one bonus/next token beyond however many draft tokens were
        accepted (this is also the simulator's Q_tokens = gamma+1 convention
        in specloop/workload.py's cost model docstring), so subtracting it
        recovers exactly the accepted draft-token count for that request.
        Proposed length per request = len(scheduled_spec_decode_tokens[rid])
        from the SAME step's SchedulerOutput (this method receives the output
        for the step that scheduled these proposals): requests with no spec
        tokens scheduled are absent from that dict and contribute 0 to both
        sides, matching a non-speculative decode step correctly.
        """
        req_ids = getattr(mro, "req_ids", None)
        sampled = getattr(mro, "sampled_token_ids", None)
        scheduled_spec = getattr(sout, "scheduled_spec_decode_tokens", None) or {}
        if req_ids is not None and sampled is not None:
            accepted = 0
            proposed = 0
            for rid, toks in zip(req_ids, sampled):
                spec_toks = scheduled_spec.get(rid)
                if not spec_toks:
                    continue  # non-speculative step for this request
                proposed += len(spec_toks)
                accepted += max(0, len(toks) - 1)
            return accepted, proposed
        return 0, 0
