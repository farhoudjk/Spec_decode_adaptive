# specloop — control-loop interference in speculative LLM serving

Experiment harness for the study of feedback-control pathologies in adaptive
speculative decoding. Simulator-backed so that the RQ2-scale factorials
(O(10^3) runs) are tractable; controller objects take a backend-agnostic
`Observation`, so the same control laws can later drive a real vLLM
integration without modification.

## Layout

| file | contents |
|---|---|
| `workload.py` | hardware/model configs, request types, arrival traces |
| `simulator.py` | step-level roofline serving simulator (spec decoding, KV pressure, chunked prefill, preemption) |
| `controllers.py` | `L_spec` and `L_admit` control laws + coordination mechanisms |
| `metrics.py` | end metrics + control-stability metrics (spectral concentration, CV, settling, cross-correlation) |
| `experiments.py` | RQ1–RQ5 drivers, ablations, mechanism isolation, regime sweep |
| `report.py` | two-column serif figures; booktabs+cellcolor LaTeX tables |
| `cli.py` | entry point |

## Running

```bash
python -m specloop.cli pilot --seeds 8          # go/no-go on the two-loop premise
python -m specloop.cli regime --seeds 4         # load x SLO regime sweep
python -m specloop.cli mechanism --seeds 6      # acceptance-heterogeneity isolation
python -m specloop.cli rq1 rq2 rq3 rq4 rq5
python -m specloop.cli abl-controller abl-loops abl-slo
python -m specloop.cli all --seeds 8
```

Outputs land in `results/{csv,fig,tab}`. Set `SPECLOOP_OUT` to redirect.
`--workers` controls process parallelism (default: cores-1).

## Cost model

Per decode step:

```
t_mem  = (W_target + KV_batch) / BW_eff
t_comp = 2 * P_active * Q_tokens / FLOPS_eff       Q_tokens = sum_i (gamma_i + 1) + prefill
t_step = max(t_mem, t_comp) + t_draft + overhead
```

Speculation raises `Q_tokens`, pushing the step from the memory-bound regime
(where speculation is nearly free) into the compute-bound regime (where it is
not). KV accounting reserves `gamma_i` slots per request *before* acceptance is
known, so speculation also consumes KV headroom; under pressure the simulator
shrinks gammas before preempting.

## Controllers

`L_spec`: `static`, `ema` (SmartSpec-style setpoint on marginal acceptance),
`bandit` (UCB over gamma arms, AdaSpec-style), `entropy` (binned predictability,
HeteroSpec-style), `dsde` (per-request ragged gamma).

`L_admit`: `static`, `slack` (slack-guided, Kairos-style), `queue`.

Coordination: `naive`, `timescale` (singular-perturbation separation),
`hysteresis` (deadband + cooldown), `mimo` (joint one-step lookahead).

These are re-implementations of the published **control laws**, not of the full
systems. Report them as such; they are not reproductions of the original
papers' end-to-end results.

## Status of the findings

Calibration runs on `llama3.2-3b/1b` so far:

**The two-loop interference premise is NOT currently supported.** Across three
regimes (loose SLO / tight SLO / saturated at rate 48), naive composition
(`D_both`) shows a larger actuation oscillation signature than the isolated
arms but **higher** goodput and SLO attainment. Oscillation without a cost is
not a result. Run `regime` over a wider grid before committing to this framing.

**A single-loop pathology does reproduce, cleanly.** With admission held
static — so nothing can be attributed to composition — a speculation controller
that senses *batch-aggregate* acceptance self-oscillates under heterogeneous
per-request acceptance:

| alpha spread | gamma CV | gamma spectral conc. | mean gamma | goodput |
|---|---|---|---|---|
| homogeneous | 0.008 | 0.003 | 1.99 | 12148 |
| heterogeneous | 0.107 | 0.799 | 1.95 | 11976 |

Same mean acceptance, same arrival process, same mean gamma and batch size —
only the per-request spread differs. Proposed channel: raising gamma drains
high-acceptance requests faster, so the residual batch is enriched in
low-acceptance requests, so the sensed batch-mean acceptance falls, so gamma
falls, so they accumulate again.

Controller dependence under heterogeneous traffic (single loop):

| L_spec | gamma spectral conc. | mean gamma | goodput |
|---|---|---|---|
| `ema` (batch-aggregate) | 0.847 | 1.95 | 11975 |
| `entropy` (batch-aggregate) | 0.860 | 3.77 | 10985 |
| `bandit` (reward-sensing) | 0.024 | 2.69 | 11510 |
| `dsde` (per-request) | 0.046 | 1.99 | 6184 |

Controllers closing the loop on a batch-mean acceptance signal limit-cycle;
those that do not, do not. `dsde` removes the oscillation but costs goodput here
because ragged gamma pays worst-case draft depth (`gmax`) while accruing only
mean-case acceptance — that specific number is the most model-dependent result
on this page and needs hardware validation before it is quoted.

## Known limitations

- Single-shot prefill approximation (chunked prefill budget is accounted in the
  step cost but a prompt completes in one step).
- Acceptance is sampled from a per-request alpha rather than from real draft/
  target distributions; the `ngram` draft mode reuses the same machinery with a
  different alpha profile rather than modelling prompt-suffix lookup directly.
- No real hardware validation. Every number here is a statement about whether a
  mechanism *can* exist in this cost model, not about its magnitude in vLLM.

## Bugs fixed during calibration (worth not reintroducing)

1. Bang-bang control laws self-oscillate in isolation, destroying attribution.
   Controllers now use setpoints with deadbands and must be shown individually
   stable before composition.
2. Rounding the actuation accumulator each step freezes a loop at small gains
   (arms C and D came out numerically identical). Controllers keep continuous
   internal state and quantize only at actuation.
3. Multiplying a setpoint by the gain then truncating drove DSDE's per-request
   gamma to 0. Gain smooths the approach to a setpoint; it never scales it.
