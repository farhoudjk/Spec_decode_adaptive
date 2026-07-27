# specloop-gpu — closed-loop speculative decoding control on vLLM

Real-GPU system for the control-loop-interference study, built as a **vLLM v1
fork via a pinned scheduler subclass**. The paper's spine: vLLM's shipped
**open-loop** `num_speculative_tokens_per_batch_size` table (k as a static step
function of batch size) versus a **closed-loop** controller that senses measured
acceptance.

## ⚠️ Verification boundary — read this first

This code was written against the documented vLLM v1 API but **was not executed
on a GPU** (the authoring environment had no CUDA/GPU/vLLM). What that means:

- **Verified, GPU-free:** controllers, telemetry, actuation path, workload
  replay logic, and analysis all pass a stubbed contract test
  (`tests/test_patch_contract.py`) that mimics the v1 scheduler override
  surface. Every Python file parses.
- **Unverified until you run it:** that the *real* vLLM field names for
  acceptance counts and KV block accounting match what the patch probes. These
  move between vLLM releases and are the first thing to check on hardware.

**Run order on your box:**
1. `pip install -r requirements.txt` then `pip install "vllm==0.9.*"` (matched to your CUDA).
2. `python tests/test_patch_contract.py` — must print `PASS` (no GPU needed).
3. Launch one arm, then walk the 5-point checklist in
   `specloop_rt/vllm_patch/PROVENANCE.md` ("What to verify on first GPU run").
   The load-bearing check is #3: `accept_rate_ema` must be in (0,1) and move
   when k moves. If it doesn't, wire the accepted-token field per
   `PROVENANCE.md#acceptance` — every controller decision depends on it.

## What touches vLLM internals

Exactly one module: `specloop_rt/vllm_patch/scheduler_patch.py`. It subclasses
the v1 `Scheduler` and overrides three methods (`schedule`,
`update_from_output`, plus a gamma-application helper). Registered via the
supported `--scheduler-cls` seam, so no upstream file is edited in place and the
patch survives rebases onto new vLLM tags. `PROVENANCE.md` maps every internal
touchpoint to its upstream `file:symbol` and documents the equivalent hard-fork
edit sites if you prefer an in-place fork.

Everything else (`controllers.py`, `interface.py`, `workload.py`,
`analysis.py`, `real_corpus.py`, `multiturn.py`) has **zero vLLM imports** and
is portable across versions.

## Layout

```
specloop_rt/
  interface.py        StepObservation / ControlAction / telemetry (no vllm)
  controllers.py      StaticTable (vLLM open-loop baseline), ClosedLoop (ours),
                      DSDE, Slack admission, Composite + coordination
  vllm_patch/         THE ONLY vllm-touching code (pinned v0.9.x v1)
    scheduler_patch.py
    PROVENANCE.md     upstream file:line for every touchpoint + first-run checklist
  real_corpus.py       ShareGPT / HumanEval / SQuAD / CNN-DailyMail loaders
  workload.py          single-shot arrival traces (step / mixed / volatile /
                       homogeneous, each with a bursty_* CV=2.5 counterpart);
                       use_real_corpus=True draws from real_corpus.py instead
                       of synthetic templates
  multiturn.py         multi-turn ShareGPT conversation replay: sequential
                       per-conversation turns with think-time gaps and a
                       growing shared-prefix prompt (needs enable_prefix_caching)
  replay.py            async single-shot trace-replay client against AsyncLLM
  analysis.py           end metrics + oscillation/settling from telemetry
                       (shared schema, works on either replay.py's or
                       multiturn.py's requests.jsonl)
configs/              per-GPU runtime configs (24GB / 48GB, single-shot /
                      multiturn)
scripts/              run_matrix.sh, mkconfig.py, analyze.py
tests/                GPU-free contract test
```

## Running an experiment

```bash
# pick the config matching your GPU
CONFIG=configs/rtx4090_24gb.yaml OUT=results_gpu RATE=8 DUR=180 SEEDS="0 1 2" \
  scripts/run_matrix.sh
```

This runs the core arms — `no_spec`, `static_spec`, `vllm_open_loop` (baseline),
`ours_closed_loop`, plus the `spec_only`/`admit_only` isolation arms — on the
mixed trace, then the coordination arms (`naive`/`timescale`/`hysteresis`) on the
perturbation trace, and aggregates into `results_gpu/summary.txt`.

Single run, synthetic prompts:

```bash
python -m specloop_rt.replay --config configs/rtx4090_24gb.yaml \
    --trace step --rate 8 --duration 180 --seed 0 --out results_gpu/probe
```

Single run, real prompts (ShareGPT/HumanEval/SQuAD/CNN-DailyMail) and/or
bursty (CV=2.5) arrivals instead of Poisson:

```bash
python -m specloop_rt.replay --config configs/rtx4090_24gb.yaml \
    --trace bursty_mixed --rate 8 --duration 180 --seed 0 \
    --real-corpus --out results_gpu/probe_bursty_real
```

`--trace` accepts `step`/`mixed`/`volatile`/`homogeneous` (Poisson, CV=1) and
their `bursty_*` counterparts (Gamma-distributed gaps, CV=2.5 by default —
see `workload.BURSTY_CV`); `--real-corpus` swaps every rtype's prompt source
from the synthetic templates to the matching real dataset
(`rag`->SQuAD, `code`->HumanEval, `chat`->ShareGPT, `reason`->CNN-DailyMail).

Multi-turn ShareGPT conversation replay — turns are submitted sequentially per
conversation (think-time gap between turns, growing shared-prefix prompt),
distinct from the independent-request model above:

```bash
python -m specloop_rt.multiturn --config configs/a6000_48gb_multiturn.yaml \
    --n-conversations 100 --rate 1.0 --min-turns 2 --max-turns 6 \
    --seed 0 --out results_gpu/multiturn_probe
```

The config passed to `multiturn` must set `model.enable_prefix_caching: true`
(see `configs/a6000_48gb_multiturn.yaml`) — each turn resubmits the full
conversation history, and prefix caching is what makes that cheap instead of
recomputing the shared prefix from scratch every turn.

## Model pairs (runtime-selectable)

Edit the `model:` block or swap config:
- `rtx4090_24gb.yaml` — Qwen2.5-7B + 0.5B draft (safe on 24GB).
- `a100_48gb.yaml` — Llama-3.1-8B + Llama-3.2-1B draft (needs 40-48GB).
- `eagle_8b.yaml` — Llama-3.1-8B + EAGLE-3 heads (self-spec, lowest memory).

70B is intentionally absent: it does not fit a single consumer card with a draft
model and KV. Use the simulator harness for anything above single-node scale;
this system is for the on-GPU end-metric and stability measurements.

## Relationship to the simulator harness

Same controller interface, same metric definitions, same trace shapes — so a
regime the simulator flags as oscillatory can be re-run here for a real
measurement, and numbers line up on axis and definition. The simulator remains
the right tool for the RQ2 gain×timescale×volatility sweep (thousands of runs);
this is the right tool for RQ3/RQ4/RQ5 on hardware.
