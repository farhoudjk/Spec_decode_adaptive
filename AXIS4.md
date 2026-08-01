# Axis-4: TTFT-predictive admission, EAGLE speculation, and load shedding

Standalone follow-up to Axis-3. Everything needed to reproduce the sweep is in
this branch; the only external dependencies are the model weights and a GPU.

**Headline result: all ten admission/γ comparisons tie the static baseline, and
the null is structural rather than statistical** — `n_finished` is bit-identical
across all five non-shedding arms in every one of the six seed×rate cells.

- `reports/axis4_predictive_eagle_report.html` — results
- `reports/axis4_methodology.html` — arm definitions, control laws, shedding accounting

---

## Why this axis exists

Axis-3's adaptive arms all tied the static baseline. Step traces showed the
reason was not that adaptation does not help, but that the controllers were
barely acting:

1. **The admission sensor could not respond proportionally.**
   `TTFTSlackAdmit` computes `slack = target_util - num_waiting/cap`, which is
   bounded above by `target_util` (0.7) and small for any queue shorter than
   the cap. Two failure modes, both observed:
   - `reason`/ngram at rate=2: `num_waiting` peaked at 37 against `cap=256`
     (ratio 0.14), so the law saw pure slack and the cap never moved — while
     TTFT p99 was 37 s against a 2.0 s SLO, a 19× breach.
   - `code`/EAGLE at rate=6: `num_waiting` peaked at 207 (ratio 0.81), so the
     law *did* cross its threshold — and still moved the cap only 256→250,
     because `slack = -0.109` gives `Δ = 0.5·8·(-0.109) = -0.43` per update.

   At that same step the predictive law computes a 138.6 s predicted wait
   against a 1.4 s budget, `slack = -98`, `Δ = -392` (clamped to the floor):
   same gain, same actuation code, ~900× the response.

2. **γ had no leverage.** TPOT p50 was identical to four decimals at γ=4 vs
   γ=1 — on ngram *and* on EAGLE, with acceptance both stable and volatile.

3. **Rates were outside the interesting regime.** Cells at rate ≥ 1.5 were in
   open-loop overload (e2e p95 ≈ the whole run length), where all admission
   policies converge by construction.

---

## What this branch adds

| Component | File | What it is |
|---|---|---|
| `TTFTPredictiveAdmit` | `specloop_rt/controllers.py` | Admission driven by a Little's-Law estimate of predicted wait vs `ttft_slo_s`, replacing the queue-to-cap ratio |
| `KVPredictiveAdmit` | `specloop_rt/controllers.py` | The above + the proportional KV ceiling, combined as `min` |
| `_PredictedWaitTerm` | `specloop_rt/controllers.py` | The sensor itself, with a trend term |
| Load shedding | `specloop_rt/replay.py` | Admission-time rejection, client-side signal |
| Offered-load accounting | `specloop_rt/analysis.py` | `slo_attainment_offered`, `shed_frac`, `n_shed`, `n_offered` |
| Split SLO metrics | `specloop_rt/analysis.py` | `ttft_attainment`, `tpot_attainment` alongside the joint AND |
| Sweep | `scripts/sweep_predictive.py` | Six bracketing arms, seed-aware summary |
| Tests | `tests/test_predictive_admit.py` | GPU-free regression tests pinning the sensor defect |
| EAGLE config | `configs/a5000_24gb_llama3_eagle.yaml` | Llama-3.1-8B + EAGLE3, `max_model_len=2048` |
| ngram config | `configs/a5000_24gb_llama3.yaml` | Llama-3.1-8B + ngram, for the `reason` rounds |

---

## Picking this up on a new machine

Everything needed to *read and extend* the work is in this branch. What is not
in it, because it is too large and is regenerable, is the environment:

| Survives a clone | Must be rebuilt on the new box |
|---|---|
| All code, configs, tests, reports | The venv (~15 GB with vllm/torch/CUDA) |
| `grid.json` for all three sweeps | The HF weight cache (~19 GB) |
| `scripts/analyze_axis4.py` | Per-cell `steps.jsonl` telemetry (~313 MB) |

So the analysis, the reports and every number quoted in them are reproducible
immediately from a fresh clone:

```bash
python3 scripts/analyze_axis4.py    # regenerates the full results table
```

Re-running the *sweep* needs the environment below, and re-running it will also
regenerate the per-cell telemetry. Note the step traces are what back the
mechanism claims (cap-vs-concurrency, gamma distribution); the committed
`grid.json` holds only per-cell summaries.

## Reproducing

### 1. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r specloop_rt/requirements.txt   # pins vllm==0.9.2, transformers==4.53.3
```

The `transformers` pin is load-bearing — see `specloop_rt/vllm_patch/PROVENANCE.md`.

### 2. Weights

`meta-llama/Llama-3.1-8B-Instruct` is gated: accept the license on its model
page, then `export HF_TOKEN=hf_...`. The EAGLE3 head
(`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`) is ungated.

Point caches at a volume with ≥ 25 GB free:

```bash
export HF_HOME=/path/to/cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
```

### 3. GPU-free checks first

```bash
python3 tests/test_predictive_admit.py   # sensor regression tests
python3 tests/test_patch_contract.py     # scheduler patch contract
```

### 4. The sweep

```bash
python3 scripts/sweep_predictive.py \
  --config configs/a5000_24gb_llama3_eagle.yaml \
  --rtypes code --rates 6 8 --seeds 0 1 2 \
  --duration 300 --out results_gpu_sweep/axis4_eagle_code
```

36 cells, roughly 6 hours on one RTX A5000. `grid.json` is written after every
cell, so the run is resumable — re-invoking skips completed cells.

---

## Constraints that will bite you

**EAGLE heads are capped at 2048 positions.** Both public EAGLE heads for
Llama-3.1-8B declare `max_position_embeddings=2048`. The `reason` workload
(CNN-DailyMail) needs up to 3043 tokens and crashes with

```
CUDA error: device-side assert triggered
index out of bounds: 0 <= idx < 2048
```

Measured worst-case totals (prompt + `max_tokens`): `reason` 3043 ✗,
`chat` 8101 ✗, `code` 1254 ✓, `rag` 994 ✓. This branch therefore runs `code`.
**Axis-4 is a new baseline, not directly comparable to the five prior `reason`
rounds.**

**Draft-model speculation does not work on vLLM 0.9.2's V1 engine.** It falls
back to V0, which then crashes `replay.py`'s hardcoded V1 `AsyncLLM`
construction. Use `spec_method: ngram` or `spec_method: eagle`.

**`GatedSpec.threshold` is a property of the substrate, not a constant.** The
setpoint law is `target = log(threshold)/log(accept_rate)`. Measured EAGLE
acceptance here spans 0.34–0.44, where the inherited `threshold=0.15` gives
γ≈1.76–2.31 — permanently below the γ=4 baseline, which would make the
comparison unfalsifiable. This branch uses **0.05** (γ≈2.78–3.65, straddling
the baseline). Re-derive it for any new model/draft-method pairing.

**`est_output_len` is workload-specific.** It is the predictive sensor's one
free parameter, set from measured `mean_output_tokens`: `code` 316,
`reason` 840. `EST_OUTPUT_LEN_BY_RTYPE` in the sweep script holds the table.

---

## Reading the results honestly

**Never report a win on `slo_attainment` when `shed_frac > 0`.** A shedding
policy graded only on admitted traffic can always win by admitting less. At
rate=8 the shedding arm reaches SLO 0.155 against a 0.017 baseline — a 9.4×
"improvement" — by rejecting 88% of arrivals and completing 276 requests
instead of 2390. On offered load it is 0.018 vs 0.017: a 0.3 SD tie. Compare
`slo_attainment_offered`.

**`slo_attainment` is noisy; report spread.** Seed SD is 0.070 at rate=6 and
0.008 at rate=8, giving a minimum detectable difference of 0.114 (127% of
baseline) and 0.012 (75%) respectively with n=3. Differences smaller than that
are not callable. The load-bearing evidence for the Axis-4 null is not the SLO
ties but the exact `n_finished` identity.

**A silent `shed_frac = 0.0` may be a bug, not a policy decision.** The first
shedding implementation read scheduler state via
`vllm_patch.last_observation()`, but vLLM v1 runs EngineCore in a **separate
process** — the scheduler set that global in the engine process while the
predicate ran in the client process and read `None` on every call. It shed
nothing and looked exactly like a legitimate tie. The predicate now uses only
client-owned state. This is why `shed_frac` is always reported.

---

## Results in this branch

`results_gpu_sweep/*/grid.json` — per-cell summary metrics, force-added past
the `results_gpu*/` ignore rule (same convention as Axis-2 and Axis-3). The
bulky per-cell `steps.jsonl`/`requests.jsonl` telemetry is **not** committed;
re-run the sweep to regenerate it.

| Directory | Contents |
|---|---|
| `axis4_eagle_code/` | 36 cells: 6 arms × rates 6/8 × 3 seeds, `code`/EAGLE3 |
| `axis3_llama3_24gb/` | 15 cells: 5 arms × rates 2/4/8, `reason`/ngram |
| `axis3_llama3_24gb_lowrate/` | 20 cells: 5 arms × rates 0.5/1/1.5/2, `reason`/ngram |

Reports in `reports/` read directly from these files.
