"""Backend-agnostic control interface and telemetry schema.

This module has NO vllm import on purpose.  It is the seam between the
version-fragile patch layer (``vllm_patch/``) and everything version-stable
(controllers, metrics, replay).  The patch layer constructs a ``StepObservation``
each scheduler step and hands it to a ``Controller``; the controller returns a
``ControlAction``.  Nothing else crosses the boundary.

Keeping this identical to the simulator's ``Observation`` is what lets the
controllers written against the simulator run unmodified on GPU.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class StepObservation:
    """Everything a controller may sense at one scheduler step.

    Populated from vLLM v1 scheduler/output state (see vllm_patch/README for the
    exact field provenance).  All quantities are per-step unless named *_ema.
    """
    t_wall: float                 # time.monotonic() at schedule() entry
    step: int
    # batch state
    num_running: int
    num_waiting: int
    num_scheduled_tokens: int     # sum over reqs this step (prefill+decode)
    num_spec_tokens: int          # scheduled speculative tokens this step
    # KV state (from KVCacheManager)
    kv_used_blocks: int
    kv_total_blocks: int
    # speculation feedback (from update_from_output of the PREVIOUS step)
    accepted_tokens: int          # total accepted across batch, last step
    proposed_tokens: int          # total proposed across batch, last step
    # rolling signals maintained by the telemetry aggregator
    accept_rate_ema: float        # accepted/proposed, EMA
    accepted_per_req_ema: float   # mean accepted length per request, EMA
    step_time_ema: float
    tpot_ema: float               # per-output-token latency, EMA
    # current actuation
    gamma_current: int
    max_num_seqs_current: int
    # SLO targets (static config, echoed for controller convenience)
    tpot_slo_s: float
    ttft_slo_s: float
    # bookkeeping for coordination mechanisms
    last_spec_action_step: int = -10 ** 9
    last_admit_action_step: int = -10 ** 9

    @property
    def kv_used_frac(self) -> float:
        return self.kv_used_blocks / max(1, self.kv_total_blocks)


@dataclass
class ControlAction:
    """What a controller may actuate. ``None`` means "leave unchanged"."""
    gamma: Optional[int] = None                 # global speculative length k
    per_request_gamma: Optional[Dict[str, int]] = None  # request_id -> k (ragged)
    max_num_seqs: Optional[int] = None          # admission cap


class Controller:
    """Base class. Real controllers live in specloop_rt.controllers."""
    name = "base"

    def on_step(self, obs: StepObservation) -> ControlAction:
        raise NotImplementedError

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Telemetry sink: append-only JSONL, one record per scheduler step.
# ---------------------------------------------------------------------------


class TelemetryWriter:
    """Low-overhead per-step logger. Buffered; flushed in the engine's own
    process so it never blocks the GPU worker."""

    def __init__(self, path: str, flush_every: int = 200, run_meta: Optional[Dict] = None):
        self.path = path
        self._buf: List[str] = []
        self.flush_every = flush_every
        self._fh = open(path, "w", buffering=1)
        if run_meta:
            self._fh.write(json.dumps({"_meta": run_meta}) + "\n")

    def record(self, obs: StepObservation, action: ControlAction) -> None:
        row = asdict(obs)
        row["act_gamma"] = action.gamma
        row["act_max_num_seqs"] = action.max_num_seqs
        row["act_ragged"] = bool(action.per_request_gamma)
        self._buf.append(json.dumps(row, separators=(",", ":")))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            self._fh.write("\n".join(self._buf) + "\n")
            self._buf.clear()

    def close(self) -> None:
        self.flush()
        self._fh.close()


class EMA:
    """Bias-corrected EMA so early steps are not pinned to the init value."""

    def __init__(self, beta: float = 0.1, init: float = 0.0):
        self.beta = beta
        self._v = 0.0
        self._c = 0.0
        self._init = init
        self._started = False

    def update(self, x: float) -> float:
        if not self._started:
            self._v = x
            self._started = True
            return self._v
        self._v = (1 - self.beta) * self._v + self.beta * x
        return self._v

    @property
    def value(self) -> float:
        return self._v if self._started else self._init
