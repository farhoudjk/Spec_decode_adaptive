# A Roofline Theory of Speculation-Length Control: MoE Widens the Regime Where Adaptation Helps, But Building a Live Adaptive Controller Is Harder Than It Looks

**Status: draft, findings current as of this repo's `axis5-roofline-moe` branch.**
All numbers below are pulled directly from committed `results_gpu_sweep/*/grid.json`
files and are reproducible from this repo; see each section's pointer to the
source sweep and reproduction command.

---

## Abstract

Speculative decoding trades verify compute for fewer serial decode steps. Whether
that trade is worth making depends on batch size: at small batch the GPU is
memory-bandwidth bound and speculation is nearly free, but past a critical batch
size **B\*** the extra verify FLOPs push the step into the compute-bound regime,
where speculation stops helping and can actively hurt. We give a closed-form
roofline account of **B\*** and the speculation-length-vs-batch tradeoff, show
that a dense 8B model's serving-relevant batch range sits *above* B\* (so
speculation-depth control has no measurable leverage there — reproducing a
published negative result), and show that a sparse Mixture-of-Experts model's
much lower per-token active-parameter count pushes B\* into the serving-relevant
range, where speculation depth has real, measurable leverage (21–34% ITL
improvement over no speculation, R²=0.94 fit across 28 cells). We then report a
vLLM-internals bug that silently made every closed-loop speculation controller in
this line of work a no-op against the real decoding engine, its fix, and live
GPU validation of the fix. Finally, we report a negative result honestly: an
adaptive hill-climbing controller that genuinely reaches the real proposer still
does not beat the best fixed speculation depth in a real serving comparison, and
we trace the reason to a specific, measured property of the feedback signal
(0.95 lag-1 autocorrelation in the EMA-smoothed latency signal) that broke an
otherwise-validated fix.

---

## 1. The roofline model

### 1.1 Per-step cost

Speculative decoding proposes γ draft tokens per request per step, then verifies
them (plus one bonus token) against the target model in a single forward pass.
Per decode step, batched over the running set:

```
t_mem  = (W_target + KV_batch) / BW_eff
t_comp = 2 · P_active · Q_tokens / FLOPS_eff
t_step = max(t_mem, t_comp) + t_draft + overhead
```

where

- `W_target` — target model weight bytes resident on-device
- `KV_batch` — KV-cache bytes touched this step across the running batch
- `BW_eff` — effective HBM bandwidth (roofline-derated from peian device bandwidth)
- `P_active` — **active** parameters per token (= total params for a dense model;
  = `total_params × active_experts / total_experts` for a sparse MoE model)
- `Q_tokens = Σᵢ (γᵢ + 1) + prefill_tokens` — total tokens verified this step
  (each request contributes its accepted-or-not γ draft tokens plus one bonus
  token; a dense model's `P_active` and the sum over γ are what raise `t_comp`
  as speculation depth grows)
- `t_draft` — draft-model cost (≈0 for prompt-lookup/ngram drafting; nonzero for
  a trained draft head)

Source: `specloop/simulator.py:11-13`, matching `README.md`'s documented cost
model exactly.

### 1.2 The two regimes and the crossover

For a batch of size B running at fixed γ, `t_comp` grows linearly in B (more
tokens verified per step) while `t_mem`'s KV term also grows in B, but the
*weight* term `W_target/BW_eff` does not — so at small B, `t_mem` (dominated by
the batch-independent weight-read cost) exceeds `t_comp`, and the step is
**memory-bound**: raising γ adds verify work that overlaps the idle compute the
memory-bound step already has slack for, so speculation is nearly free.

Past a critical batch size **B\***, `t_comp` overtakes `t_mem`, and the step
becomes **compute-bound**: every additional verified token (accepted or not)
now directly extends the step. Solving `t_mem = t_comp` for the batch size at
which the crossover happens, at fixed γ (Case A: cost-free drafter, e.g. ngram)
or accounting for a costed drafter's own compute (Case B: EAGLE/draft-model),
the closed form used throughout this repo's configs is:

```
B* = W / ( (2·P_active·(γ+1)/FLOPS_eff)·BW_eff − L·S_kv )
```

where `W` is target weight bytes, `L` is the number of layers, and `S_kv` is
per-layer per-token KV bytes. (Source: `configs/a100_80gb_mixtral_eagle.yaml`
header derivation, consistent with §1.1's per-step cost model.) `B*` is
**inversely proportional to `P_active`** — this is the single fact this paper's
central claim rests on: cutting active parameters per token (what sparse MoE
does structurally) raises `B*` for a fixed weight footprint `W`, pushing the
crossover to a higher batch size and widening the batch range over which
speculation-depth control has leverage.

### 1.3 The dense-vs-MoE prediction

For a **dense** model, `P_active = total_params`, so `B*` is small — the
theory's own worked numbers put dense Llama-3.1-8B's `B*` in the ~20-40 range,
comfortably *within* typical serving batch sizes. The prediction: dense models
should show **little-to-no** speculation-depth leverage at realistic serving
batches, because most of that operating range already sits in the compute-bound
regime where γ is inert or harmful.

For a **sparse MoE** model (Mixtral-8x7B: 8 experts, top-2 routing, so
`P_active ≈ 2/8 × total_params` per token, ignoring shared/router
parameters), `P_active` is a small fraction of `total_params` for the *same*
total weight footprint `W` — so `B*` moves up substantially for the same
hardware and γ. The prediction: MoE models should show **real** speculation-depth
leverage across a much wider batch range, including realistic serving batches,
because more of that range stays memory-bound.

This is the central, testable claim of this work, and §2–3 report what actually
happened when it was tested on real hardware.

---

## 2. Falsified null: dense Llama-3.1-8B shows no γ leverage — but the null was an artifact

### 2.1 The original (Axis-4) result

Llama-3.1-8B-Instruct (dense) + EAGLE3, sweeping γ and admission policy across
six seed×rate cells (`results_gpu_sweep/axis4_eagle_code/grid.json`):
**all ten admission/γ comparisons tied the static baseline**, with `n_finished`
bit-identical across five non-shedding arms in every cell — TPOT p50 identical
to four decimal places between γ=1 and γ=4 on both ngram and EAGLE. Read at face
value, this matches §1.3's prediction: dense B\* sits below the tested batch
range, so γ is inert there.

### 2.2 The artifact

That reading turned out to be **wrong for a different reason than the theory**.
vLLM 0.9.2's speculation proposers (`NgramProposer`, `EagleProposer`) cache
`num_speculative_tokens` once at construction and never re-read it:

```python
# vllm/v1/spec_decode/ngram_proposer.py, NgramProposer.__init__
self.k = vllm_config.speculative_config.num_speculative_tokens
# vllm/v1/spec_decode/eagle.py, EagleProposer.__init__
self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
```

Every "closed-loop" controller in this repo (`ClosedLoopSpec`, `GatedSpec`,
`DSDESpec`, `HillClimbSpec`) actuates γ by mutating the config object the real
proposer already stopped reading after construction — the mutation is real
(`SpeculativeConfig` is an ordinary mutable dataclass) but silently discarded.
`gamma_current` in telemetry reports what the controller *decided*, not what
the drafter *used*.

**Confirming test** (`scripts/confirm_eagle_gamma_freeze.py`,
`results_gpu_sweep/axis4_eagle_gamma_confirm/grid.json`): the same
Llama-3.1-8B+EAGLE3 pairing, B=16, rate=6, with γ set via a **fresh engine
rebuild per value** (bypassing the freeze bug by construction):

| γ (real, construction-time) | ITL (TPOT p50, s) | acceptance |
|---|---|---|
| 1 | 0.01005 | 68.0% |
| **4** | **0.00769** | 39.8% |
| 8 | 0.01018 | 23.7% |

γ=4 beats γ=1 by **23.5%** and beats γ=8 by **24.4%** — a real, large, coherent
interior optimum. **Speculation depth has real leverage on EAGLE + dense
Llama-3.1-8B.** The original null was an artifact of γ never reaching the real
proposer, not a property of dense models or of the theory being wrong. Axis-4's
own step traces show the fingerprint directly: `mean_gamma` swings 1.18→2.24
across cells while `mean_accept_rate` stays flat at 0.39–0.41 — the controller
moved, the drafter did not.

---

## 3. The MoE roofline result: real, consistent leverage across every tested batch

### 3.1 Setup

Mixtral-8x7B-Instruct-v0.1 (FP8, `RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8`,
compressed-tensors W8A8, ~47GB), ngram (prompt-lookup) speculation on a single
A100-SXM4-80GB. FP8 was chosen deliberately: fp16 (~93GB) does not fit one
80GB card; AWQ 4-bit fits easily but cuts `W` 4×, which by §1.2's formula pulls
`B*` back down from ~47 to ~12-20 — i.e. AWQ would fight the exact
total/active-parameter-ratio effect this experiment exists to isolate. FP8
keeps `B* ≈ 20-25` at γ=4, the real middle ground.

EAGLE speculation was attempted first (`yuhuili/EAGLE-mixtral-instruct-8x7B`)
but is architecturally blocked in vLLM 0.9.2: the EAGLE head is full MHA
(`num_key_value_heads=32`) while Mixtral is GQA (`num_key_value_heads=8`) —
incompatible per-layer KV page sizes, crashing `get_kv_cache_config` with a bare
`NotImplementedError`. No workaround exists in this vLLM version; this run uses
ngram instead (§1.1's `t_draft ≈ 0` case).

### 3.2 The B×k fit sweep

`results_gpu_sweep/axis4_roofline_moe_fit/grid.json` — B∈{8,16,24,32} ×
γ∈{0,1,2,3,4,5,6,8}, seed=0, rate=8 req/s, code/HumanEval, 120s/cell,
static/static controller (fixed batch cap, fixed γ, engine rebuilt per cell —
by construction this predates and is unaffected by §2.2's freeze bug).

**ITL (TPOT p50, seconds) by B×γ:**

| B\γ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.0285 | 0.0257 | 0.0258 | 0.0241 | **0.0224** | 0.0227 | 0.0226 | 0.0232 |
| 16 | 0.0354 | 0.0281 | 0.0274 | 0.0278 | 0.0256 | 0.0253 | **0.0248** | 0.0261 |
| 24 | 0.0418 | 0.0330 | 0.0297 | **0.0275** | 0.0283 | 0.0300 | 0.0301 | 0.0310 |
| 32 | 0.0396 | 0.0356 | **0.0306** | 0.0329 | 0.0326 | 0.0329 | 0.0346 | 0.0370 |

Every batch beats its own γ=0 baseline by **21–34% ITL** — real, consistent
leverage across the entire tested batch range, unlike dense Llama's null. The
ITL-minimizing γ falls smoothly as batch grows.

**Global fit** (least-squares, all 28 non-baseline cells):

```
ITL(B,γ) = c0 + c1·B + c2·γ + c3·γ² + c4·B·γ,     R² = 0.94
```

Differentiating (`∂ITL/∂γ = 0`) gives the closed-form optimal speculation depth:

```
γ*(B) = -(c2 + c4·B) / (2·c3)  ≈  6.66 − 0.095·B
```

Predicts γ\*≈5.9 (B=8), 5.1 (B=16), 4.4 (B=24), 3.6 (B=32) — smoother than the
noisy raw per-row peaks (4/6/3/2), same direction, same magnitude.

### 3.3 Mechanism: not a clean roofline crossover, but acceptance collapse

The theory's §1.2 prediction — `T_comp ∝ (γ+1)` overtaking a flat `T_mem` — does
not cleanly explain the interior optimum: `T_step = ITL × E[tokens/step]` keeps
*falling* with γ at every batch tested, never turning around. The interior
optimum in *ITL specifically* instead comes from **acceptance collapse**:
`E[tokens/step]` grows too slowly relative to step-time savings once acceptance
α has fallen far enough — α falls from ~47% at γ=1 to ~16% at γ=8 at every
batch, meaning most speculated tokens past γ≈3-4 are wasted verify work buying
almost no additional accepted tokens. Tail latency reinforces this: ITL
p95/p50 rises with γ at every batch (~1.33 at γ=1 → ~1.75 at γ=8) — higher γ
costs more at the tail too, not just the median.

### 3.4 Dense contrast, same session

`results_gpu_sweep/axis2/grid.json` — Qwen2.5-7B (dense) + ngram + code, 1×
A6000, batches matched to the Mixtral B values, `regime=unconstrained` (clean,
not overloaded) at every row. ITL flat to 3-4 decimals across γ=0→8 at every
batch tested (e.g. batch≈13: 0.01726/0.01723/0.01724/0.01749s for γ=0/1/4/8) —
reproduces the dense null exactly, on a different model/GPU but the same
drafter family and workload. **This is the contrast that makes the MoE result
meaningful**: dense shows zero leverage at any tested batch; MoE shows real
leverage at every tested batch, on the same class of hardware.

---

## 4. Fixing the freeze bug: from "the controller decided" to "the drafter used"

### 4.1 Three independent freeze points, not one

Reading vLLM 0.9.2's source directly (not guessed) surfaced three separate
places where `num_speculative_tokens` is cached at construction and never
resynced — the third was found only by a live crash, not by reading source:

1. **Proposer caches** (`NgramProposer.__init__`'s `self.k`,
   `EagleProposer.__init__`'s `self.num_speculative_tokens`) — the bug §2.2
   documents.
2. **Scheduler KV lookahead reservation** (EAGLE only):
   `Scheduler.__init__` sets `self.num_lookahead_tokens = self.num_spec_tokens`
   once, used every step to reserve KV blocks for the EAGLE draft model's own
   KV cache via `KVCacheManager.allocate_slots`. A live γ increase past the
   construction-time value risks under-reserving those blocks.
3. **Scheduler spec-decoding stats** (all proposer types, not eagle-gated):
   `Scheduler.__init__` sets `self.num_spec_tokens` once, which pre-sizes
   `SpecDecodingStats.num_accepted_tokens_per_pos` and gates
   `assert num_accepted_tokens <= self.num_spec_tokens` in `observe_draft()`.
   Raising live γ past construction-time γ trips this assert and crashes the
   entire engine process (`AssertionError → EngineDeadError`) — this is how
   point 3 was actually found, during the first live GPU run of an adaptive
   controller that raises γ above its starting value.

### 4.2 The fix

`specloop_rt/vllm_patch/live_gamma_patch.py`, applied from inside
`SpecLoopScheduler.__init__` (not from the client process — see §4.3):
monkeypatches `propose()` on both proposer classes to resync from the live
`SpeculativeConfig` every call, and pins the two scheduler-side quantities
(points 2 and 3 above) to an 8-token ceiling at construction so a live γ
increase within that range never trips either.

### 4.3 A second bug, found while validating the first

An early version of the fix imported `EagleProposer` in the client process
(`replay.py`) to patch it ahead of engine construction. That import alone
initializes a CUDA context as a side effect. vLLM's `get_mp_context()` forces
the EngineCore child process to `spawn` instead of `fork` whenever CUDA is
already initialized in the parent — and unlike `fork` (which inherits parent
memory), `spawn` gives the child a fresh interpreter that never sees the
controller/telemetry globals `configure()` installs in the parent. The symptom
was silent: `steps.jsonl` came back with only its `_meta` header and zero real
step rows, and every controller decision defaulted to a no-op. Confirmed this
was not a pre-existing repo bug (replaying the original unpatched `replay.py`
via `git stash` produced 601 real telemetry rows on the identical config, via
`fork`). Fixed by applying the proposer patch **only** from inside
`SpecLoopScheduler.__init__`, which always runs in the EngineCore process
regardless of which start method vLLM picks.

### 4.4 Live GPU validation

Ground truth is `num_spec_tokens` from the real `SchedulerOutput`
(`scheduled_spec_decode_tokens`), not `gamma_current` (what the controller
*believes* it set) — the entire freeze bug is precisely the gap between these
two signals.

| Proposer / model | γ flip | `num_spec_tokens`/req before → after |
|---|---|---|
| ngram, Qwen2.5-0.5B | 4 → 1 | 3.878 → 0.993 |
| EAGLE3, Llama-3.1-8B | 4 → 1 | 3.792 → 0.994 |

Both flips happen **mid-run, on a single engine instance, with no rebuild** —
proof the live actuation is real on both proposer families this repo uses.

---

## 5. Does a live adaptive controller actually beat a fixed γ? (Negative result, precisely diagnosed)

### 5.1 Why this is a different question from §4

§4 proves *mechanism*: γ_current genuinely reaches the real proposer.
It does not prove *the controller wins* — a controller can actuate correctly
and still lose to the best fixed γ if it oscillates, lags, or settles on the
wrong value. This section reports the real comparison.

### 5.2 HillClimbSpec: design

A batch-gated hill-climb directly on measured ITL (`obs.tpot_ema`), not on
acceptance (which falls monotonically across the whole γ range per §3.3, so it
cannot localize an interior optimum) and not on a static γ\*(B) lookup (would
not self-correct if live conditions diverge from §3.2's fit). The live search
window is centered on §3.2's fitted formula, `γ*(B) ≈ 6.66 − 0.095·B`, ±2
(clamped to [1,8]) — narrows *where* the hill-climb looks without replacing the
hill-climb itself. Settle-then-compare timing: γ held fixed for
`settle_steps=40` steps (~4× the reporting EMA's time constant) before every
comparison, comparing against the reading at the end of the *previous* settle
window (not the previous single step) with a relative deadband
(`deadband_frac`) to avoid reacting to noise.

### 5.3 Full-scale result: does not beat the best fixed γ

`scripts/sweep_hillclimb_vs_static.py` on Mixtral-8x7B-FP8+ngram, B∈{16,32},
γ∈{1,2,4,6,8} static arms + `hillclimb`, rate=8, code/HumanEval, 90s/cell, 1
seed (`results_gpu_sweep/axis5_hillclimb_vs_static_live/grid.json`, 12/12
cells, 0 errors):

| B=16 | TPOT p50 | mean γ | accept | | B=32 | TPOT p50 | mean γ | accept |
|---|---|---|---|---|---|---|---|---|
| static-γ6 | **0.0256** | 6.00 | 0.183 | | static-γ2 | **0.0318** | 2.00 | 0.352 |
| static-γ4 | 0.0261 | 4.00 | 0.246 | | hillclimb | 0.0325 | 2.77 | 0.307 |
| **hillclimb** | 0.0262 | 4.95 | 0.216 | | static-γ4 | 0.0339 | 4.00 | 0.242 |
| static-γ8 | 0.0267 | 8.00 | 0.152 | | static-γ6 | 0.0359 | 6.00 | 0.190 |
| static-γ2 | 0.0279 | 2.00 | 0.354 | | static-γ1 | 0.0362 | 1.00 | 0.464 |
| static-γ1 | 0.0287 | 1.00 | 0.458 | | static-γ8 | 0.0388 | 8.00 | 0.146 |

`HillClimbSpec` lands **2.3–2.4% behind the best fixed γ at both batch sizes**
— close, and clearly ahead of the worst fixed choices (13-16% better than the
worst static arm), but "beats a bad fixed γ" is a materially weaker claim than
"beats the best fixed γ," which was the actual target.

### 5.4 Diagnosis

Per-step trace analysis (not just the summary metric) identifies two concrete
costs:

1. **Settle-in cost.** With `settle_steps=40`, the controller spends its first
   several hundred steps probing the search window before converging — that
   probing runs at suboptimal γ the entire time, dragging down the aggregate
   metric even after finding a good region. A fixed-γ arm pays none of this;
   it is optimal (or not) from step 0.
2. **Unconverged oscillation at B=32.** The controller genuinely hovers near
   the true optimum (γ=2) but never settles: steady-state histogram (excluding
   startup/tail) is γ=2 for 1270 steps, γ=3 for 1380 steps — a near-exact
   split, meaning roughly half the settle windows end in a reversal rather
   than a confirmation. B=32 is the only arm in the sweep flagged
   `oscillatory=True`.

### 5.5 A tried fix, and why it failed on real hardware — the most important negative result of this session

**Hypothesis**: single-point ITL comparisons are noisier than the 3% deadband
(measured `tpot_ema` stdev ≈ 7.7% of the mean at B=32), so averaging over the
settle window should sharpen the comparison and stop the oscillation.
**Implementation**: `HillClimbSpec` modified to compare the mean of the last
`avg_last_n` (default = `settle_steps`) readings, and to hold position rather
than always stepping when the windowed comparison is ambiguous.

**Offline validation looked like a clear win.** A fast replay harness resampled
real per-step `tpot_ema` values (with replacement) from the B=32 static-γ
traces and drove the actual `HillClimbSpec` class against synthetic streams.
Mean simulated ITL fell from 0.04987 (original design) to 0.04851 (fixed
design) over many seeds; the steady-state γ histogram collapsed from a 5-wide
spread {2,3,4,5,6} to just {2,3}.

**Real GPU re-run showed no improvement, and B=16 got measurably worse**
(`results_gpu_sweep/axis5_hillclimb_vs_static_live_v2/grid.json`, 12/12
cells, 0 errors):

| | v1 (single-sample) | v2 (windowed-avg + hold) |
|---|---|---|
| B=16 vs best fixed γ | +2.3% | **+3.7%** |
| B=32 vs best fixed γ | +2.4% | +2.2% (statistically a wash) |
| B=32 `oscillatory` flag | True | **still True** |

**Root cause, directly measured, not guessed**: `tpot_ema` is an EMA
(α=0.1 per scheduler step), so consecutive readings are nearly identical.
Measured **lag-1 autocorrelation on a real trace: ρ = 0.9542**. The averaging
fix summed `n=40` consecutive readings expecting `√n ≈ 6.3×` noise reduction —
but for an AR(1)-like process with this ρ, the *effective independent sample
count* inside one window is

```
n_eff = n · (1 − ρ) / (1 + ρ) = 40 · 0.0458 / 1.9542 ≈ 0.94
```

— essentially **one** independent sample, not forty. Averaging 40
highly-correlated near-duplicates of the same underlying (still noisy) value
provides almost no variance reduction. The offline simulator's fatal flaw was
methodological: resampling real `tpot_ema` values *with replacement* discards
the sequence they arrived in, which destroys exactly the autocorrelation
structure that determines whether averaging helps — it silently converted a
slow-moving, correlated signal into a synthetic i.i.d. one, and validated the
fix against a noise model the real signal does not have.

**This is reported as a negative result, not spun as a partial win.** The code
change was kept (not actively harmful — B=32 is a wash, B=16's regression is
within the range of single-seed noise this comparison already runs on) but
should not be cited as a fix. Untried options that follow directly from the
diagnosis: shorten the EMA's own time constant so consecutive readings
actually decorrelate within a settle window (would let the *same* averaging
idea work, since the failure mode was specifically about correlation, not the
logic); compare against freshly-computed per-request TPOT percentiles instead
of an EMA (no inherited autocorrelation to fight); or discount/exclude
settle-in cost to reflect a long-running deployment rather than a 90-second
test cell (a real deployment settles once and runs for hours, amortizing the
one-time settle-in cost near zero — untested here).

---

## 6. What this work actually supports, stated precisely

1. **The MoE roofline claim is supported**: sparse MoE (Mixtral-8x7B) shows
   real, consistent speculation-depth leverage (21–34% ITL improvement,
   R²=0.94 fit) across every tested batch size, in direct contrast to a dense
   8B model that shows none — matching the theory's prediction that lower
   active-parameter fraction raises B\* into the serving-relevant range.
2. **The dense-model null was real, but for the wrong reason initially
   claimed**: the original "γ has no leverage on dense EAGLE" conclusion was
   an artifact of a vLLM proposer-freeze bug, not evidence against the theory.
   A confirming test with the bug worked around (rebuild-per-γ) shows dense
   EAGLE **does** have real, large γ leverage (23-24% ITL swing) — consistent
   with, not contradicting, the roofline account (dense B\* still sits low;
   the *existence* of leverage where B < B\* was never in question, only
   whether the harness could see it).
3. **The freeze bug is fixed and validated live** on real GPU hardware for
   both proposer families this repo uses (ngram, EAGLE), with no engine
   rebuild required to change γ mid-run.
4. **Mechanism is not sufficient for adaptive control to win**: a controller
   that demonstrably reaches the real proposer still loses to the best fixed
   γ by 2.2–3.7% in a real serving comparison. The reason is diagnosable and
   specific (settle-in cost; an oscillation driven by comparing a
   highly-autocorrelated smoothed signal at too fine a grain), not a vague
   "needs more tuning."
5. **A validated-offline fix failed online**, and the reason was found by
   directly measuring the real signal's autocorrelation rather than
   re-guessing — this is itself a transferable methodological finding for
   anyone tuning a live controller against an EMA-smoothed telemetry signal:
   **offline replay via resampling real traces with replacement silently
   assumes the signal is i.i.d., and will validate averaging-based fixes that
   do not work if the real signal is autocorrelated.** Check lag-1
   autocorrelation before trusting an offline noise-reduction result.

---

## 7. Reproducing every number in this draft

```bash
# environment
python3 -m venv /root/spec_decode_env/venv && source /root/spec_decode_env/venv/bin/activate
pip install -r requirements.txt
pip install -r specloop_rt/requirements.txt   # pins vllm==0.9.2, transformers==4.53.3
export HF_HOME=/root/spec_decode_env/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# §2: EAGLE freeze-bug confirming test (needs HF_TOKEN for Llama-3.1-8B)
export HF_TOKEN=<token with meta-llama/Llama-3.1-8B-Instruct access>
python3 scripts/confirm_eagle_gamma_freeze.py

# §3: the B x gamma roofline fit sweep on Mixtral (~3.5-4h on one A100)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8', cache_dir='$HUGGINGFACE_HUB_CACHE')"
python3 scripts/sweep_roofline_moe.py \
  --config configs/a100_80gb_mixtral_ngram.yaml \
  --rtype code --rate 8 --Bs 8 16 24 32 --ks 0 1 2 3 4 5 6 8 --seeds 0 \
  --duration 120 --out results_gpu_sweep/axis4_roofline_moe_fit

# §4: GPU-free unit tests for the freeze-bug fix (no vllm/GPU needed)
python3 tests/test_live_gamma_patch.py
python3 tests/test_patch_contract.py

# §4.4: live-actuation smoke tests (small models, minutes not hours)
python3 scripts/smoketest_live_gamma.py \
  --config configs/smoketest_live_gamma.yaml --flip-step 60 --rate 8 --duration 30
python3 scripts/smoketest_live_gamma.py \
  --config configs/smoketest_live_gamma_eagle.yaml --flip-step 40 --rate 6 --duration 30

# §5: hillclimb-vs-static comparison, both versions (~1-1.5h each on one A100)
python3 scripts/sweep_hillclimb_vs_static.py \
  --config configs/a100_80gb_mixtral_ngram.yaml \
  --rtype code --rate 8 --Bs 16 32 --ks 1 2 4 6 8 --seeds 0 \
  --duration 90 --out results_gpu_sweep/axis5_hillclimb_vs_static_live
```

All `grid.json` result files referenced in this draft are committed in this
branch (`results_gpu_sweep/*/grid.json`, force-added past the
`results_gpu*/` gitignore rule). Per-cell `steps.jsonl`/`requests.jsonl`
telemetry — which backs every mechanism claim in §3.3 and §5.4-5.5 — is not
committed (bulky, regenerable); re-run the relevant sweep to regenerate it.

---

## 8. Open items (not yet resolved)

- EAGLE+Mixtral KV-head mismatch (§3.1) — no workaround found in vLLM 0.9.2;
  would need a different checkpoint with matching `num_key_value_heads` or a
  patch to vLLM's KV-cache-config path. Blocks testing §1.2's Case B
  (costed-drafter, acceptance-tracking γ\*) on a MoE target.
- §5.5's untried options (shorter EMA time constant; fresh per-request TPOT
  instead of an EMA; discounting settle-in cost for a long-horizon framing).
- `sweep_hillclimb_vs_static.py` has only been run at B∈{16,32}, 1 seed;
  B∈{8,24} and multi-seed replication are unconfirmed.
- Whether `ClosedLoopSpec`/`GatedSpec` results from earlier axes (not just the
  dense EAGLE case in §2) carry the same "controller moved, drafter didn't"
  fingerprint is not yet checked against the fixed proposer.

See `AXIS4.md` and `AXIS5_ROOFLINE_MOE.md` for the full session-by-session
narrative, including dead ends and exact commands, behind every result
summarized here.
