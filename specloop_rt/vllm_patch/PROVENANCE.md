# vLLM patch provenance

Every place `scheduler_patch.py` touches vLLM internals, keyed to the upstream
API this was written against. **Pinned target: vLLM v0.9.x, v1 engine.** These
internals move; when something breaks, this file tells you which upstream symbol
to re-check.

> The patch was written against the documented v1 API and **not executed on a
> GPU**. The first real run is the integration test. Run
> `tests/test_patch_contract.py` against a stubbed scheduler first (no GPU
> needed) to catch signature drift before launching the engine.

## First real-hardware finding: pin `transformers`, not just `vllm`

`pip install vllm==0.9.2` alone pulls the newest `transformers` (observed:
5.14.1 in July 2026), because vLLM's own metadata puts no upper bound on it.
That newer transformers already registers an `"aimv2"` AutoConfig entry;
`vllm/transformers_utils/configs/ovis.py` tries to register the same key and
raises `ValueError` at import time. That import sits underneath
`vllm.v1.core.sched.scheduler.Scheduler`, so it takes down the scheduler
import too -- caught by this file's `try/except`, so `SpecLoopScheduler` ends
up subclassing plain `object` instead of the real `Scheduler`
(`SpecLoopScheduler.__mro__` shows this directly; `_VLLM_OK` is `False`).
`SpecLoopScheduler.__init__` does raise on this, so it fails loudly at engine
construction rather than silently producing bad numbers -- but the fix is
one level removed from the traceback you'll see, so it's worth having here.

Fix: `pip install transformers==4.53.3` (or another version contemporary
with vLLM 0.9.2, mid-2025) after installing vllm. See
`specloop_rt/requirements.txt`.

## Registration

vLLM v1 supports a pluggable scheduler class via `SchedulerConfig.scheduler_cls`
(config key `--scheduler-cls`), documented at
`vllm.config.scheduler` → "The scheduler class to use … Can be a class directly
or the path to a class of form `mod.custom_class`."

We register `specloop_rt.vllm_patch.SpecLoopScheduler`. This is the
minimal-diff form of a fork: no upstream file is edited in place, so it survives
`git rebase` onto new vLLM tags. If you want a **hard fork** instead, the three
overrides map to these in-place edit sites:

| override | upstream file:symbol (v0.9.x) |
|---|---|
| `schedule()` | `vllm/v1/core/sched/scheduler.py :: Scheduler.schedule()` (docs: v1 core.py:315 / sched/scheduler.py:322) |
| `update_from_output()` | `vllm/v1/core/sched/scheduler.py :: Scheduler.update_from_output()` |
| gamma application | proposer under `vllm/v1/spec_decode/` + `SpeculativeConfig.num_speculative_tokens` |

## <a name="admission"></a>Admission (max_num_seqs)

- Read by the base `schedule()` when bounding the running set.
- Lives on `self.scheduler_config.max_num_seqs` (`vllm.config.scheduler`).
- We mutate it **before** `super().schedule()`. This is the least-invasive
  admission actuator. A stricter fork would instead cap inside the waiting→running
  promotion loop in `schedule()`.
- Watermark interaction: `SchedulerConfig` also has a free-KV watermark used when
  admitting; our cap composes with it (we never raise above the hard config max).

## <a name="kv"></a>KV usage

- `self.kv_cache_manager` (constructed in `Scheduler.__init__`, docs
  v0.9.0 sched/scheduler.py).
- Block accounting attribute names have churned across v0.9.x. We probe, in order:
  `num_gpu_blocks`, `block_pool.num_gpu_blocks`; free via
  `block_pool.free_block_queue.num_free_blocks`, then `block_pool.num_free_blocks`,
  then `num_free_blocks`.
- If all probes miss, we report `(0, 1)` (kv_used_frac≈0) — check here first if
  KV-pressure controllers behave oddly.

## <a name="tpot"></a>TPOT proxy

- **Second real-hardware finding**, found the same way as the acceptance-field
  bug below: the original formula (`step_time_ema / (accepted_tokens_this_step
  + num_running)`) computes aggregate batch throughput (tokens/sec across
  every running request that step), not per-request TPOT (seconds between one
  request's own consecutive tokens). It was silently wrong by 15-300x at every
  concurrency actually tested (num_running 20-32 and 215-256; see
  `results_gpu_sweep/axis3_validate*/grid.json`), and the error *grows* with
  concurrency because the old denominator scaled with `num_running` while true
  per-request TPOT does not.
- Fix (current code): `tpot_ema` tracks `step_time_ema` directly. In v1's
  continuous batching each running request advances by ~1 decode token per
  scheduler step (barring preemption), so one step's wall time already
  approximates one request's inter-token gap -- no division needed.
- Consequence while this was wrong: `GatedSpec`'s decode-bound gate
  (`specloop_rt/controllers.py`) compares `tpot_ema` to `tpot_slo_s` and
  stayed closed even when the system was genuinely decode-bound by 6x its
  SLO. `SlackAdmit` reads the same signal and was affected identically. If a
  closed-loop controller looks unresponsive to real decode pressure, this is
  the first thing to re-verify -- log true per-request TPOT (from
  `requests.jsonl`, independent of this EMA) alongside `tpot_ema` for one run
  and confirm they track.

## <a name="acceptance"></a>Acceptance counts

- The load-bearing signal for every closed-loop spec controller.
- Reported per step in `model_runner_output`. Exact field differs by version;
  candidates probed: `num_accepted_tokens` + `num_spec_tokens`, then
  `spec_token_ids` (proposed only).
- **If accepted count is unavailable in your version**, telemetry still logs
  *proposed*; wire the accepted count here. In v1 the rejection sampler returns
  the accepted length per request — surface that tensor onto
  `model_runner_output` in the worker and sum it. This is the single most
  important field to verify on first run; a wrong wiring makes `accept_rate_ema`
  meaningless and every controller decision invalid.

## <a name="k"></a>Speculative token count (gamma / k)

- **Global k**: `SpeculativeConfig.num_speculative_tokens`
  (`vllm.config.SpeculativeConfig`; docs "Speculative Decoding"). We write the
  effective k each step. Note the base scheduler caps
  `max_num_batched_tokens` accounting for appended spec tokens
  (`vllm.config.scheduler`: "can be smaller … such as speculative decoding") —
  raising k raises the per-step token budget, which is exactly the coupling under
  study.
- **Ragged per-request k**: the `DynamicProposer` / `eagle_dynamic` path
  (PR #26504) reads a per-request k. We stash `_specloop_per_req_k` on the
  proposer object; the hard-fork version reads it inside the proposer's
  propose() loop.
- vLLM's shipped **open-loop** table `num_speculative_tokens_per_batch_size`
  (`[start_bs, end_bs, k]`) is what `StaticTableSpec` reimplements as the paper's
  baseline. Confirm your build accepts that key if you also want to run the
  in-engine open-loop policy directly for cross-checking.

## What to verify on first GPU run (in order)

1. Import succeeds → version matches the pin.
2. `test_patch_contract.py` passes against the stub → override signatures match.
3. `accept_rate_ema` in telemetry is in (0,1) and tracks k changes → acceptance
   wiring correct (see #acceptance).
4. `kv_used_frac` rises with concurrency → KV wiring correct (see #kv).
5. Changing the controller changes `act_gamma`/`act_max_num_seqs` in telemetry
   AND changes measured throughput → actuation is really applied, not just logged.
