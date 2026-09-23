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

---

# Real-hardware MoE speculative decoding: B0–B4

Measurement on live SGLang + EAGLE3 speculative decoding against two MoE
targets (Qwen3-30B-A3B and gpt-oss-20b), across a 12-point (batch admission
cap `B` × arrival rate `λ`) load grid.

## The five policies (B0–B4)

| | policy | what it does |
|---|---|---|
| **B0** | static default | one fixed `(D,W)` tree shape, same at every load point |
| **B1** | architecture-blind budget | greedy marginal-utility picker fit on a `D+W`-only cost model |
| **B2** | expert-footprint budget (this work) | same greedy picker, cost model adds the fitted expert-activation (`E`) term |
| **B3** | SGLang's shipped adaptive controller | `--speculative-adaptive`, `D`-switching on `ema_accept_len` |
| **B4** | EcoSpec | cost-aware draft-candidate selection within a fixed tree, greedy `P(t)/ΔCost(t\|buffer)` (arXiv 2607.12696) |

B4 needs `W ≥ 2` (W=1 has no branching to select over). B4's baseline uses
`P(t)=1` for every candidate and ground-truth (not predicted) routing,
reported as an oracle upper bound rather than a live-serving policy.

## Prerequisites

- An SGLang server environment (`sglang[all]`, `ninja`) with a CUDA toolkit
  matching the driver — see `scripts/rebuild_env_gptoss.sh`.
- `CUDA_HOME` set to that toolkit's path.
- For gpt-oss-20b: the published EAGLE3 draft (`nebius/EAGLE3-gpt-oss-20b`)
  needs its config patched before SGLang can load it.
  `rebuild_env_gptoss.sh` downloads it, patches it, and prints the
  `GPTOSS_DRAFT_PATH` to export. Qwen needs no patch (its published draft,
  `lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex`, loads as-is).
- gpt-oss-20b requires `--moe-runner-backend triton --mem-fraction-static 0.94`
  on every launch.

## Running each policy

All scripts live in `scripts/` and write to `results_gpu_sweep/<name>/`.
Each is idempotent to rerun and safe to background (`setsid ... & disown`)
for a multi-hour unattended grid.

```bash
export CUDA_HOME=/mnt/data/venv/lib/python3.12/site-packages/nvidia/cu13
export HF_HOME=/mnt/data/hf_cache               # optional, defaults shown

# gpt-oss-20b only, once per fresh environment:
scripts/rebuild_env_gptoss.sh
export GPTOSS_DRAFT_PATH=/mnt/data/eagle3-gptoss-20b-sglang   # printed by the rebuild script
```

**B0 / B1 / B2 / oracle**

```bash
scripts/run_qwen_4x3.sh
python3 scripts/pick_tree.py --B 8 --rate 2

scripts/run_gptoss_4x3.sh
python3 scripts/compute_gptoss_b0_b1_b2.py
```

**B3**

```bash
scripts/run_qwen_adaptive_4x3.sh
scripts/run_gptoss_adaptive_4x3.sh
```

**B4 (EcoSpec)**

```bash
scripts/run_ecospec_gptoss_4x3.sh
```

**End-to-end**

```bash
scripts/chain_gptoss_full.sh
```

## Reading the output

- `results_gpu_sweep/<grid>/grid.json` — raw per-`(D,W,B,rate)` cell:
  throughput, acceptance length, `mean_distinct_experts`.
- `*/b0_b1_b2_full.json` (gpt-oss) or `pick_tree.py`'s stdout (Qwen) —
  per-load-point shape choice and goodput for B0/B1/B2/oracle.
- `*_adaptive_4x3/result.json` — per-load-point static-arm sweep and the
  adaptive arm's settled depth, switch count, and goodput (B3).
- `ecospec_4x3/summary.json` — per-load-point mean distinct-experts under
  confidence-only vs. EcoSpec selection, and the resulting reduction (B4).

## Limitations

- B4 reports an expert-footprint reduction, not a throughput number.
- B4's candidate pool is treated as unordered within each verify step
  rather than as a tree; this likely understates its real effect.
- B4 is evaluated on gpt-oss-20b only.
- gpt-oss's rate values are not numerically comparable to Qwen's; both are
  chosen to keep each model decode-bound rather than queue-bound.
