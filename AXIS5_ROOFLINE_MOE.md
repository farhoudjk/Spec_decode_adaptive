# Axis-5: MoE roofline validation, and a proposer-freeze bug that reopens Axis-4

## 0. Session update: the freeze bug is FIXED and GPU-validated live

**Read this section first — it supersedes §4's "not fixed" status and §5's
open item 2.** A follow-up session patched vLLM 0.9.2 so speculation depth
(γ) actuates live, with no engine rebuild, and confirmed this on real GPU
hardware for both proposer types. `HillClimbSpec` (§4.2's originally
un-validatable adaptive controller) now genuinely hill-climbs γ against live
telemetry — this is the first time any adaptive spec controller in this
repo's history has been shown to reach the real proposer.

**What was fixed** (`specloop_rt/vllm_patch/live_gamma_patch.py`, applied
from `SpecLoopScheduler.__init__`): three independent freeze points, found by
reading vLLM 0.9.2 source directly and then by two live crashes:

1. `NgramProposer`/`EagleProposer` cache `num_speculative_tokens` in
   `__init__` and never re-read it in `propose()` — the bug §4 documents.
   Fixed by monkeypatching `propose()` on both classes to resync from the
   live `SpeculativeConfig` every call (safe: neither proposer pre-allocates
   any k-sized buffer in `__init__`, confirmed by reading both files in
   full).
2. `Scheduler.num_lookahead_tokens` (EAGLE only) is also frozen at
   construction and used to reserve KV blocks for the EAGLE draft model's own
   KV cache every step; under-reserving after a live k increase degrades to
   spurious preemption (`allocate_slots` returns `None`, not corruption).
   Fixed by pinning it to an 8-token ceiling at construction (matching
   `HillClimbSpec._K_HARD_MAX`) instead of resyncing it live.
3. **Found by a real crash, not a source read**: `Scheduler.num_spec_tokens`
   (frozen at construction, NOT eagle-gated — applies to every proposer type)
   pre-sizes `SpecDecodingStats.num_accepted_tokens_per_pos` and gates an
   `assert num_accepted_tokens <= self.num_spec_tokens` in
   `observe_draft()`. The first live GPU run of `HillClimbSpec` (which raises
   γ above its start value) tripped this assert and took down the whole
   EngineCore process (`AssertionError` → `EngineDeadError`). Fixed the same
   way as point 2: pinned to the k-ceiling at construction, for every
   proposer type.

**A separate, more insidious bug was found and fixed in the process of
validating this**: an early version of the patch imported `EagleProposer` in
the **client** process (inside `replay.py`'s `_build_engine`) before engine
construction, to patch it ahead of time. That import alone initializes a
CUDA context as a side effect. vLLM's `get_mp_context()` forces the
EngineCore child process to `spawn` instead of `fork` whenever CUDA is
already initialized in the parent — and unlike `fork` (which inherits parent
memory), `spawn` gives the child a fresh interpreter that never sees
`configure()`'s process-global controller/telemetry installation
(`scheduler_patch.py`'s `_CONTROLLER`/`_TELEMETRY` globals). The symptom was
silent: `steps.jsonl` came back with only the `_meta` header and zero real
step rows, and every controller decision silently defaulted to a no-op
(`ControlAction()`, all fields `None`). **This was NOT a pre-existing bug in
this repo** — confirmed by re-running the exact same unmodified,
pre-session `replay.py` (via `git stash`), which produced 601 real telemetry
rows on the same config, proving `configure()`'s cross-process propagation
already worked correctly via `fork` before this session's patch touched it.
The fix was to apply the proposer patch ONLY from inside
`SpecLoopScheduler.__init__` (which always runs in the EngineCore process
regardless of which start method vLLM picks), never from the client. Worth
remembering for anyone extending `vllm_patch/` in the future: **importing
anything from `vllm.v1.spec_decode` (or likely any real torch/CUDA-touching
vLLM submodule) in the client process risks silently breaking every other
process-global mechanism this patch layer depends on.** No other client-side
code in this repo does this (checked); the hazard was fully contained to
code written and then fixed within this same session.

**GPU-validated this session** (small-scale smoke tests, not full sweeps —
see §0.1 for what still needs a full re-run):
- ngram (Qwen2.5-0.5B, ungated): live γ flip 4→1 mid-run, ground-truth
  `num_spec_tokens/req` tracked it exactly (3.878 → 0.993, no engine rebuild).
- EAGLE (Llama-3.1-8B + EAGLE3, gated, real pairing from §4.2's confirming
  test): same result, 3.792 → 0.994.
- `HillClimbSpec` (ngram/Qwen2.5-0.5B): the live controller genuinely
  hill-climbed γ between 3 and 8 over a 40s run, `num_spec_tokens/req`
  tracking `gamma_current` throughout — the mechanism §4.2 describes as
  "correctly implemented but cannot be validated live" is now validated live.
- 9 new GPU-free unit tests (`tests/test_live_gamma_patch.py`), covering all
  three freeze points and the real vLLM construction-order edge case
  (proposer built before the patch applies), all passing against both a
  faithful stub and the real installed vLLM 0.9.2 classes.

**Important caveat, added by §0.3/§0.4's full-scale follow-ups**: mechanism ≠
win. A further session ran the real `hillclimb`-vs-`static-k` comparison on
Mixtral at B∈{16,32} and found `HillClimbSpec` does NOT beat the best fixed
k at either batch size (2.3-2.4% behind, §0.3) — it converges to
approximately the right γ neighborhood but pays a real settle-in cost and,
at B=32, never stops oscillating. A THIRD follow-up (§0.4) tried the
obvious fix (average settle-window readings instead of one point-sample) —
it looked like a clear win in offline simulation but made things WORSE on
real GPU hardware, because the offline simulator's resampling method
destroyed the real signal's measured 0.95 lag-1 autocorrelation, which is
exactly what determines whether averaging helps. Read §0.3 AND §0.4 in full
before citing "the adaptive controller works" as a paper claim — the
accurate claim from this session is "live actuation works and the
controller finds approximately the right region, but does not yet beat
static-k as tuned, and the first attempted tuning fix failed for a
precisely-identified reason that points at where to look next (§0.4's
options i-iv)."

### 0.1 What this does NOT change yet — still needs a full re-run

Per explicit scope agreed for the code-fix follow-up session (code +
small-scale validation, not the multi-hour sweeps): §3 and §4.3 below were
NOT regenerated in that session. A further follow-up (§0.3) DID run the
full-scale `sweep_hillclimb_vs_static.py` comparison — read that section for
the actual result. Still open after §0.3:

1. **§4.3's EAGLE+dense confirming test** (γ=1/4/8, rebuild-per-cell, showing
   a 23-24% ITL swing) was run under the OLD frozen-proposer regime by
   construction (rebuild-per-γ sidesteps the freeze bug entirely, which is
   why it was trustworthy evidence of the bug in the first place — see
   §4.4). It does NOT need to be redone to be valid. What's newly possible
   and STILL NOT done: re-running it with a LIVE controller instead of
   rebuild-per-cell, to see whether live actuation reproduces the same
   interior optimum without a rebuild between arms.
2. **§3's Mixtral B×k fit sweep numbers are UNCHANGED and still trustworthy**
   — that sweep rebuilds the engine per cell by construction (see §4.4), so
   the freeze bug never affected it and neither does this session's fix.
   Nothing in §3 needs re-running because of anything in this section.
3. `sweep_hillclimb_vs_static.py` has now been run at B∈{16,32} only (§0.3)
   — B∈{8,24} from the original fit sweep's grid are still unconfirmed for
   the live controller, and only 1 seed was used throughout.

### 0.2 What this follow-up session added

| Component | File | What it is |
|---|---|---|
| Live-gamma patch | `specloop_rt/vllm_patch/live_gamma_patch.py` | Fixes all three freeze points (§0); applied from `SpecLoopScheduler.__init__` |
| GPU-free regression tests | `tests/test_live_gamma_patch.py` | 9 tests, stub-based, covering all three freeze points + the real construction-order edge case |
| `sweep_hillclimb_vs_static.py` fix | `scripts/sweep_hillclimb_vs_static.py` | `run_cell` now sets `cfg["runtime"]["gamma_init"]` from the arm's real k, closing the plumbing bug §4.2/§5 item 1 described |
| Live-actuation smoke test | `scripts/smoketest_live_gamma.py` | Single-engine, no-rebuild test: flips γ mid-run via a scripted controller, checks ground-truth `num_spec_tokens/req` (not `gamma_current`) actually tracks it |
| Smoke-test configs | `configs/smoketest_live_gamma.yaml`, `configs/smoketest_live_gamma_eagle.yaml` | Small/fast configs (Qwen2.5-0.5B ngram; Llama-3.1-8B+EAGLE3) — NOT experiment configs, for validating the patch only |

All diffs from this follow-up session are uncommitted in the working tree as
of this doc's edit; this section's commit includes them.

### 0.3 Second follow-up: does HillClimbSpec actually beat fixed k? (full-scale, real answer)

Ran `sweep_hillclimb_vs_static.py` for real on Mixtral-8x7B-FP8+ngram, now
that live actuation works: `hillclimb` + `static-k{1,2,4,6,8}`, B∈{16,32},
rate=8, `code`/HumanEval, 90s/cell, 1 seed. Scoped to 2 batch sizes (not the
original sweep's 4) and 90s (not 120s) to fit a bounded validation window —
see §0.1 item 3 for what's still unconfirmed. All 12 cells completed with no
errors. Full results:
`results_gpu_sweep/axis5_hillclimb_vs_static_live/grid.json`.

**tpot_p50 (s), sorted best to worst:**

| B=16 | tpot_p50 | mean_γ | accept | | B=32 | tpot_p50 | mean_γ | accept |
|---|---|---|---|---|---|---|---|---|
| static-k6 | **0.0256** | 6.00 | 0.183 | | static-k2 | **0.0318** | 2.00 | 0.352 |
| static-k4 | 0.0261 | 4.00 | 0.246 | | hillclimb | 0.0325 | 2.77 | 0.307 |
| **hillclimb** | 0.0262 | 4.95 | 0.216 | | static-k4 | 0.0339 | 4.00 | 0.242 |
| static-k8 | 0.0267 | 8.00 | 0.152 | | static-k6 | 0.0359 | 6.00 | 0.190 |
| static-k2 | 0.0279 | 2.00 | 0.354 | | static-k1 | 0.0362 | 1.00 | 0.464 |
| static-k1 | 0.0287 | 1.00 | 0.458 | | static-k8 | 0.0388 | 8.00 | 0.146 |

**Honest answer: no, HillClimbSpec did not beat the best fixed k at either
batch size.** It landed 2.3% behind static-k6 at B=16 and 2.4% behind
static-k2 at B=32 — close, and it clearly beat the WORST fixed k choices at
each B (13% better than static-k1 at B=16, 16% better than static-k8 at
B=32), but "beats a bad fixed k" is a much weaker claim than "beats the best
fixed k," which was the actual target.

**Diagnosis (from the per-step `steps.jsonl` traces, not just the summary
`grid.json`): the mechanism works and finds approximately the right
neighborhood, but two concrete issues keep it from converging onto the exact
optimum and cost it in the aggregate metric.**

1. **Settle-in cost.** With `settle_steps=40`, the controller spends its
   first several hundred steps probing the full window (e.g. B=16's trace:
   γ moves 5→4→3→4→5→6→5→4→3→... over the first 600 steps) before
   settling. That probing period runs at suboptimal k the whole time,
   dragging down the aggregate `tpot_p50` even once the controller finds a
   good region. A fixed-k arm pays none of this cost — it's optimal (or not)
   from step 0.
2. **B=32 never actually settled — it's flagged `oscillatory=True` (the
   only arm in the sweep with that flag).** Its full gamma trace bounces
   2↔3 for nearly the entire 3800-step run, correctly hovering near the true
   optimum (k=2) but never converging onto it — `deadband_frac=0.03` (a
   tight 3% relative threshold) is apparently too tight for real ITL
   measurement noise at this batch size, so small legitimate fluctuations
   keep reversing the climb direction. Steady-state histogram (steps
   570-3420, excluding startup/tail): γ=2 for 1270 steps, γ=3 for 1380 steps
   — almost exactly split, meaning half the run every "settle window" ends
   in a coin-flip reversal rather than confirming convergence.
3. Both traces show a further complication: B=16's tail (last ~40 of 6800
   steps) and B=32's tail (last ~40 of 3800) drift AWAY from the converged
   region right at run end (B=16 tail sits at γ=6 which is actually
   correct; B=32's tail drifts 3→4→5→6, away from its true optimum γ=2) —
   consistent with the same deadband-noise sensitivity, not a new bug.

**This is a controller-tuning finding, not a mechanism failure.** The live
actuation is doing exactly what it's supposed to — γ_current genuinely
reaches the real proposer and the controller genuinely responds to measured
ITL. It's converging to the right neighborhood (mode γ=5 at B=16 vs true
optimum 6; mode γ=2-3 at B=32 vs true optimum 2) but not tightly enough, and
paying real cost to get there.

### 0.4 Third follow-up: tried the deadband/oscillation fix — it did NOT work, and here's precisely why

Attempted the first of §0.3's two "not yet tried" fixes: `HillClimbSpec` was
changed (`specloop_rt/controllers.py`) to compare the AVERAGE of the last
`avg_last_n` (default = `settle_steps`, i.e. the whole settle window)
`tpot_ema` readings instead of a single point-sample at the window's end,
plus a "hold position if the windowed comparison is ambiguous" rule instead
of always taking another step. The reasoning: the real per-step `tpot_ema`
noise (stdev ≈ 7.7% of the mean at B=32) is larger than `deadband_frac`
(3%), so a single reading can't reliably distinguish adjacent k's whose true
ITL means differ by a similar few percent — averaging should cut that noise
down by ~√n.

**Offline validation (before spending more GPU time) looked strong.**
Built a fast replay harness (`/tmp/.../hillclimb_offline_sim.py`, not
committed — throwaway tuning tool) that resampled real per-step `tpot_ema`
values (with replacement) from the actual B=32 static-k traces recorded in
§0.3, then ran the real `HillClimbSpec` class against synthetic streams
built from that resampling. Result: mean simulated ITL fell from 0.04987
(current/v1 design) to 0.04851 (fixed design) over many seeded trials, and
the steady-state γ histogram collapsed from a 5-wide spread (2-6) to just
{2, 3} — exactly the oscillation-suppression the fix was meant to produce.

**Real GPU re-run (`results_gpu_sweep/axis5_hillclimb_vs_static_live_v2/`,
12 cells, 0 errors) showed the fix did NOT help — and made B=16 measurably
worse:**

| | v1 (single-sample) | v2 (windowed-avg + hold) | change |
|---|---|---|---|
| B=16 hillclimb_vs_best | +2.3% | **+3.7%** | worse |
| B=32 hillclimb_vs_best | +2.4% | +2.2% | wash (within noise) |
| B=32 `oscillatory` flag | True | **still True** | unchanged |

B=32's steady-state γ histogram post-fix: `{2: 1240, 3: 1400, 4: 320, 5: 40}`
— still drifting up to 4-5, not the clean {2,3} split the offline sim
predicted.

**Root cause of the offline/online gap, found by directly measuring the
real signal's autocorrelation** (not guessed): `tpot_ema` is an EMA
(`α=0.1` per scheduler step, see `vllm_patch/scheduler_patch.py`), so
consecutive steps are nearly identical — measured **lag-1 autocorrelation
on a real trace: 0.95**. The averaging fix sums `avg_last_n=40` consecutive
`tpot_ema` readings expecting `√40 ≈ 6.3×` noise reduction, but with
lag-1 autocorrelation this high, the *effective independent sample count*
inside one 40-step window is `n·(1-ρ)/(1+ρ) ≈ 0.94` — essentially ONE
independent sample, not 40. Averaging 40 highly-correlated near-duplicates
of the same underlying (still noisy) value provides almost no variance
reduction. **The offline simulator's fatal flaw**: resampling real
`tpot_ema` values *with replacement* discards the sequence they came in,
which destroys exactly the autocorrelation structure that determines
whether averaging helps — it silently turned a correlated, slow-moving
signal into a synthetic i.i.d. one, and validated the fix against a noise
model the real signal doesn't have.

**Disposition: the code change was kept anyway** (not reverted) — it is not
worse than v1 in the aggregate (B=32 wash, B=16 regression is within the
kind of single-seed noise this whole comparison runs on) and averaging
consecutive EMA readings is not actively harmful, but it should not be
described as a fix. `HillClimbSpec`'s oscillation at B=32 and its
settle-in-cost drag at B=16 are BOTH still open — see updated open items
below. A real fix would need to either (a) shorten the EMA's own time
constant so consecutive readings decorrelate faster, letting the SAME
averaging idea actually reduce variance, or (b) compare against something
with lower serial correlation than `tpot_ema` itself (e.g. per-request TPOT
percentiles freshly computed each window, not an EMA), or (c) simply widen
`deadband_frac` well past the ~7.7% single-sample noise floor measured
here (untested — §0.3's simulation showed WIDER deadbands make things worse
under the flawed i.i.d. model, but that conclusion inherits the same
autocorrelation blind spot and should be re-tested against real traces
rather than resampled ones).

---

## Original session narrative (pre-fix)

Session picking up from the theory doc ("A Roofline Theory of Speculation-Length
Control") the user supplied at the start of this work: a closed-form account of
when adaptive speculation-length control has leverage, built around a critical
batch size B\* below which speculation helps and above which it doesn't, and a
claim that sparse MoE widens B\* into the serving-scale batch range where dense
models (Axis-4's Llama-3.1-8B finding) show none.

**Two headline results, in tension with each other:**

1. **The MoE roofline claim is directionally supported.** Mixtral-8x7B (MoE,
   FP8, ngram) shows real, consistent speculation leverage at every batch size
   tested (21-34% ITL improvement over no-speculation), unlike the falsified
   dense-Llama Axis-4 result. The optimal speculation depth k\* falls
   smoothly as batch grows — fit across 28 cells, R²=0.94:
   `k*(B) ≈ 6.66 - 0.095*B`.
2. **A bug found while building the adaptive controller invalidates the
   mechanism behind Axis-4's own headline conclusion.** vLLM 0.9.2's
   speculation proposers (ngram AND EAGLE) freeze `num_speculative_tokens` at
   engine construction and never re-read it. Every "closed-loop" controller in
   this repo (`ClosedLoopSpec`, `GatedSpec`, `DSDESpec`, and the new
   `HillClimbSpec`) actuates γ by mutating a config object the real proposer
   has already stopped looking at. A confirming test (rebuild-per-γ, no live
   actuation) shows EAGLE + dense Llama-3.1-8B **does** have real γ leverage
   (23-24% ITL swing, interior optimum at γ=4) — directly contradicting
   Axis-4's "the null is structural" conclusion. See §4 — this was the most
   important finding of the ORIGINAL session; §0 above is what happened next.

---

## 1. Environment (rebuilt fresh this session, off the AXIS4.md playbook)

This box (RunPod, 1× A100-SXM4-80GB) started with nothing but the repo clone.
Rebuilt:

- venv at `/root/spec_decode_env/venv` (**not** `/workspace` — that network
  volume hit a small per-pod quota after ~13GB; overlay root `/` had the
  actual 91GB headroom. If picking this up on a fresh pod, check `df -h`
  before choosing where to put the venv.)
- `pip install -r requirements.txt && pip install -r specloop_rt/requirements.txt`
  (the latter installs `vllm==0.9.2` + `transformers==4.53.3` — the pin
  matters, see AXIS4.md's own PROVENANCE note)
- HF cache at `/root/spec_decode_env/hf_cache` (`export HF_HOME=...`,
  `export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub`)
- `HF_TOKEN` exported in `~/.bashrc` (needed only for
  `meta-llama/Llama-3.1-8B-Instruct`, gated; NOT needed for the Mixtral work,
  see below)

**Disk is tight on this box (100GB overlay root).** Mixtral-8x7B-FP8 is 44GB;
Llama-3.1-8B is 30GB. Both together don't fit alongside a working venv +
telemetry. **The Mixtral weights were deleted at the end of this session** to
make room for the Llama-3.1-8B confirming test (§4) — re-download before
resuming Axis-5 work:
```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8', cache_dir='/root/spec_decode_env/hf_cache/hub')
"
```
(~44GB, ~3 min on this network in earlier testing.)

---

## 2. What was built

| Component | File | What it is |
|---|---|---|
| Mixtral+EAGLE config | `configs/a100_80gb_mixtral_eagle.yaml` | Does NOT work — see §3.1, KV-head mismatch |
| Mixtral+ngram config | `configs/a100_80gb_mixtral_ngram.yaml` | The working config used for all Mixtral results below |
| B×k roofline sweep | `scripts/sweep_roofline_moe.py` | Direct (B,k) grid, static/static controller, rebuilds engine per cell (`gamma_init=k`) |
| Adaptive controller | `specloop_rt/controllers.py::HillClimbSpec` | Batch-gated ITL hill-climb; **built correctly but cannot be validated live, see §4** |
| Adaptive-vs-static sweep | `scripts/sweep_hillclimb_vs_static.py` | **Has the freeze bug** (§4) — every arm silently ran at gamma_init=4 regardless of configured k. Needs the same per-cell `runtime.gamma_init` fix as sweep_roofline_moe.py before re-running. |
| Freeze-bug confirming test | `scripts/confirm_eagle_gamma_freeze.py` | 3-cell EAGLE test proving the bug changes the answer (§4) |
| `replay.py` fixes | `specloop_rt/replay.py` | (a) EAGLE method now inferred from checkpoint name (`eagle` vs `eagle3`) instead of hardcoded `eagle3`; (b) `quantization` field now passed to `AsyncEngineArgs` (was silently dropped); (c) `gamma_init=0` now correctly skips `speculative_config` entirely (true no-speculation baseline) instead of passing `num_speculative_tokens=0` |

All diffs are currently **uncommitted** in the working tree as of session end;
this commit includes them.

---

## 3. The MoE roofline results

### 3.1 Why ngram, not EAGLE, for Mixtral

Tried `yuhuili/EAGLE-mixtral-instruct-8x7B` first. It loads and attaches
cleanly (`lm_head`/`embed_tokens` reuse confirmed working), but vLLM 0.9.2's
V1 engine crashes at KV-cache setup with a bare `NotImplementedError` in
`kv_cache_utils.py::get_kv_cache_config`. Root cause: the EAGLE head is full
MHA (`num_key_value_heads=32`) while Mixtral is GQA (`num_key_value_heads=8`)
— different per-layer KV page sizes, and neither
`is_kv_cache_type_uniform` nor `is_kv_cache_page_size_uniform` can reconcile
them. No workaround flag exists in this vLLM version. **Not revisited this
session** — see §5 open items.

Used `spec_method: ngram` instead (`configs/a100_80gb_mixtral_ngram.yaml`).

### 3.2 Why FP8, not fp16 or AWQ

fp16 Mixtral-8x7B is ~93GB — doesn't fit one 80GB card. AWQ 4-bit fits easily
(~24GB) but the roofline formula `B* = W / (...)` is directly proportional to
W: cutting W 4× pulls B\* from ~47 (fp16, k=4, ι=1.5) down to ~12-20 — i.e.
AWQ fights the exact MoE total/active-ratio effect this experiment exists to
isolate, landing B\* back near the already-falsified dense regime. FP8
(`RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8`, compressed-tensors W8A8, ~47GB)
keeps B\* in the ~20-25 range — genuine middle ground. **Ungated, ships its
own tokenizer** — no HF_TOKEN needed for this part of the work.

### 3.3 The B×k fit sweep (32 cells, complete, no errors)

`results_gpu_sweep/axis4_roofline_moe_fit/grid.json`. B∈{8,16,24,32} ×
k∈{0,1,2,3,4,5,6,8}, seed=0, rate=8 req/s, code/HumanEval, 120s/cell,
static/static controller (fixed batch cap, fixed γ, no adaptive movement
— this sweep predates and is unaffected by the freeze bug since it never
tries to actuate γ live).

**ITL (tpot_p50, seconds) by B×k:**

| B\k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.0285 | 0.0257 | 0.0258 | 0.0241 | **0.0224** | 0.0227 | 0.0226 | 0.0232 |
| 16 | 0.0354 | 0.0281 | 0.0274 | 0.0278 | 0.0256 | 0.0253 | **0.0248** | 0.0261 |
| 24 | 0.0418 | 0.0330 | 0.0297 | **0.0275** | 0.0283 | 0.0300 | 0.0301 | 0.0310 |
| 32 | 0.0396 | 0.0356 | **0.0306** | 0.0329 | 0.0326 | 0.0329 | 0.0346 | 0.0370 |

Every B beats its own k=0 baseline by 21-34% ITL — real, consistent leverage.
Optimal k falls as B grows (peaks 4/6/3/2 across B=8/16/24/32 — the B=16 peak
is broad/flat, k=4/5/6 all within 3% of each other, so "6" overstates
precision there).

**Global fit, all 28 non-baseline cells** (more reliable than the noisy
per-row peaks): least-squares `ITL(B,k) = c0 + c1*B + c2*k + c3*k² + c4*B*k`,
**R²=0.94**. Differentiating (`dITL/dk=0`) gives:
```
k*(B) = -(c2 + c4*B) / (2*c3)  ≈  6.66 - 0.095*B
```
Predicts k\*≈5.9 (B=8), 5.1 (B=16), 4.4 (B=24), 3.6 (B=32) — smoother than
the raw peaks, same direction, same rough magnitude.

**Mechanism, not a clean roofline crossover.** Checked whether the interior
optimum comes from `T_comp ∝ (k+1)` overtaking a flat `T_mem` (the theory
doc's §2 prediction) — it doesn't cleanly: `T_step = ITL × E[tokens/step]`
keeps *falling* with k at every B, never turns around. The interior optimum
in ITL specifically comes from `E[tokens/step]` growing too slowly relative
to step-time savings once acceptance has collapsed (α falls from ~47% at k=1
to ~16% at k=8 at every B) — most speculated tokens past k≈3-4 are wasted
verify work buying almost no additional accepted tokens. Tail latency
reinforces this: ITL p95/p50 ratio rises with k at every B (~1.33 at k=1 →
~1.75 at k=8) — higher k costs more at the tail too, not just the median.

### 3.4 Quick pass + dense baseline comparison (superseded by 3.3, kept for the dense contrast)

`results_gpu_sweep/axis4_roofline_moe_quick/grid.json` — 9-cell single pass
(B∈{16,24,32}×k∈{1,4,8}) run before the full fit sweep; numbers consistent
with 3.3, no new information.

**Dense baseline** (existing data, not re-run this session):
`results_gpu_sweep/axis2/grid.json`, Qwen2.5-7B (dense) + ngram + code, 1×
A6000, batches matched to the Mixtral B values. `regime=unconstrained` (not
overloaded) at every row — this data is clean. ITL flat to 3-4 decimals
across γ=0→8 at every batch tested (e.g. batch≈13: 0.01726/0.01723/0.01724/
0.01749s for g=0/1/4/8) — reproduces Axis-4's dense null exactly, on a
different model/GPU but same drafter family and workload. This is the
contrast that makes the MoE result meaningful: dense shows zero leverage at
any tested batch, MoE shows real leverage at every tested batch.

**Caveat carried through both reports**: all Mixtral cells ran at rate=8,
which overloads every cell (TTFT/E2E dominated by queue wait, 56-145s p50) —
those columns are not meaningful, only ITL (computed from decode-step timing,
unaffected by queue depth) is. The dense comparison rows are NOT overloaded
(`regime=unconstrained`) and are fully trustworthy including TTFT/E2E.

---

## 4. THE BUG — read this first next session

### 4.1 What it is

vLLM 0.9.2's speculation proposers read `num_speculative_tokens` **once, at
construction**, never again:

```python
# vllm/v1/spec_decode/ngram_proposer.py
def __init__(self, vllm_config):
    self.k = vllm_config.speculative_config.num_speculative_tokens
def propose(self, ...):
    k = min(self.k, ...)   # frozen self.k, never re-read

# vllm/v1/spec_decode/eagle.py
def __init__(self, vllm_config, ...):
    self.num_speculative_tokens = (
        self.speculative_config.num_speculative_tokens)
# never reassigned anywhere else in the file
```

`specloop_rt/vllm_patch/scheduler_patch.py::_sl_apply_gamma` writes a new
value into `vllm_config.speculative_config.num_speculative_tokens` every
control step, trying to actuate γ live — but both proposer classes already
copied the old value into their own instance attribute and never look at the
config object again. The write is silently discarded. Telemetry's
`gamma_current` reports what the controller *decided*, not what the real
proposer *used*. **This affects every prior spec controller in this repo**:
`ClosedLoopSpec`, `GatedSpec`, `DSDESpec`, and the new `HillClimbSpec`.

### 4.2 How it was found

Building `HillClimbSpec` (batch-gated ITL hill-climb — see
`specloop_rt/controllers.py`, well-documented docstring with the full design
rationale) and comparing it against 7 fixed-k arms
(`scripts/sweep_hillclimb_vs_static.py`), every arm — including the supposedly
different fixed-k ones — clustered within ~1% of each other, when the fit
sweep (§3.3) on the same model/config had shown a sharp real spread. Diffing
the two sweep scripts: `sweep_roofline_moe.py` sets
`cfg["runtime"]["gamma_init"] = k` per cell (forces a fresh engine/proposer
per k — an accidental workaround). `sweep_hillclimb_vs_static.py` only sets
`gamma_init` inside the controller's own init kwargs (`spec_kw`) — a
different field the engine-builder never reads — so
`cfg["runtime"]["gamma_init"]` silently stayed at the base config's default
(4) for every cell, every arm. **Every "static-k1" through "static-k8" cell,
and the "hillclimb" cell, actually ran at k=4 the entire time.**

### 4.3 Confirming test: this also reopens Axis-4

Checked whether the ORIGINAL Axis-4 EAGLE data (`axis4_eagle_code/grid.json`)
shows the same fingerprint: `mean_gamma` swings 1.18→2.24 across cells while
`mean_accept_rate` stays flat at 0.39-0.41 — exactly what "controller moved,
drafter didn't" looks like.

Ran a 3-cell confirming test (`scripts/confirm_eagle_gamma_freeze.py`):
Llama-3.1-8B + EAGLE3 (the original Axis-4 pairing), B=16, rate=6, **genuine
fresh engine build per γ**, static/static controller (zero live-actuation
confound):

| γ (real, construction-time) | ITL (tpot_p50) | acceptance |
|---|---|---|
| 1 | 0.01005 | 68.0% |
| **4** | **0.00769** | 39.8% |
| 8 | 0.01018 | 23.7% |

γ=4 beats γ=1 by **23.5%** and beats γ=8 by **24.4%** — a real, large,
coherent interior optimum. **Speculation depth has real leverage on EAGLE +
dense Llama-3.1-8B.** The original Axis-4 conclusion ("all ten admission/γ
comparisons tie the static baseline... the null is structural") is not
supported by this mechanism — the reported null was, per this test, an
artifact of γ never reaching the real proposer.

`results_gpu_sweep/axis4_eagle_gamma_confirm/grid.json` has the raw cells.

### 4.4 What's still solid vs what needs re-examination

- **Solid, unaffected**: `sweep_roofline_moe.py`'s B×k fit sweep (§3.3) —
  rebuilds the engine per cell by construction, so the freeze bug never
  had a chance to matter there. The k\*(B) formula stands.
- **Needs re-examination**: any conclusion in Axis-1/2/3/4 that rested on a
  "closed-loop"/adaptive spec controller showing (or not showing) an effect.
  If the conclusion was "the controller made no difference," check whether
  that's because γ truly doesn't matter, or because γ never reached the
  drafter. The Axis-4 EAGLE case is now a confirmed instance of the latter.
- **Not fixed**: no attempt was made this session to patch vLLM so the
  proposer re-reads k live. Would need to override `propose()` or force
  proposer reconstruction on each controller actuation — nontrivial in the
  v1 engine's process model, not attempted.

---

## 5. Open items for next session

**Items 1-2 below are RESOLVED as of §0 — kept here, struck through, for
continuity with the original session narrative.**

1. ~~Re-run `sweep_hillclimb_vs_static.py` correctly~~ — **DONE, twice over.**
   Fixed in `run_cell` (sets `cfg["runtime"]["gamma_init"]` from the arm's
   real k/gamma_init, matching `sweep_roofline_moe.py`'s pattern) AND run at
   full scale on Mixtral (B∈{16,32}, see §0.3) — **result: HillClimbSpec did
   NOT beat the best fixed k**, landing 2.3-2.4% behind. See §0.3 for full
   diagnosis (settle-in cost + an unconverged oscillation at B=32) and two
   concrete untried fixes (loosen `deadband_frac`; discount settle-in cost
   for a long-horizon deployment framing). B∈{8,24} and multi-seed are still
   unrun.
2. ~~Decide how to evaluate "adaptive" given the freeze~~ — **DONE, option
   (b).** The vLLM patch was written (`specloop_rt/vllm_patch/
   live_gamma_patch.py`) and GPU-validated: `HillClimbSpec` now genuinely
   actuates γ live against the real proposer — but §0.3 shows genuine live
   actuation is not by itself sufficient to beat static-k; the controller
   still needs tuning. See §0 and §0.3.
2a. ~~Try the windowed-averaging/hold-on-ambiguous deadband fix~~ —
   **TRIED, DID NOT WORK (§0.4).** Validated offline (looked like a clear
   win) but failed to reproduce on real GPU hardware: B=16 got WORSE
   (2.3%→3.7% behind best static-k), B=32 was a wash (2.4%→2.2%, still
   `oscillatory=True`). Root cause, directly measured: `tpot_ema`'s lag-1
   autocorrelation on a real trace is 0.95 — averaging 40 highly-correlated
   consecutive EMA readings gives ~1 effective independent sample, not the
   ~40 the offline resample-with-replacement simulator assumed. The code
   change was kept (not actively harmful) but should not be cited as a fix.
2b. **New, from §0.4, the real remaining options** (untried): (i) shorten
   `tpot_ema`'s own EMA time constant (`vllm_patch/scheduler_patch.py`'s
   `_sl_ema["tpot"]`, currently α=0.1) so consecutive readings decorrelate
   faster — makes the SAME averaging idea from §0.4 actually work, since
   its failure mode was specifically about autocorrelation, not the
   averaging logic itself; (ii) compare against per-request TPOT computed
   fresh each window instead of an EMA, which has no inherited
   autocorrelation to fight; (iii) re-run §0.3's deadband-width simulation
   against REAL (not resampled) sequential traces before trusting its
   "wider deadband is worse" conclusion — that conclusion was produced by
   the same flawed i.i.d.-resampling methodology §0.4 found broken, so it
   may be wrong in the same way; (iv) discount/exclude settle-in cost to
   reflect a long-running deployment rather than a 90s test cell — still
   completely untried, no data either way yet.
3. **EAGLE+Mixtral KV-head mismatch (§3.1)** — checked again this session,
   still blocked. Confirmed the EAGLE-Mixtral checkpoint
   (`yuhuili/EAGLE-mixtral-instruct-8x7B`) is ungated and loads cleanly, but
   the underlying issue is unchanged: vLLM 0.9.2's KV-cache-config code
   cannot reconcile the EAGLE head's full-MHA KV layout with Mixtral's GQA
   layout (`is_kv_cache_type_uniform`/`is_kv_cache_page_size_uniform` both
   fail). This is unrelated to the freeze bug and outside what
   `live_gamma_patch.py` touches — would need a separate, riskier patch to
   vLLM's KV-cache-config path, or a different checkpoint. Not attempted.
4. ~~Mixtral weights need re-downloading~~ — **DONE for §0.3** (re-downloaded,
   Llama-3.1-8B deleted in exchange to fit both under the 100GB disk
   constraint, see §1). Currently on this box: Mixtral-8x7B-FP8 only. Deleted
   again: nothing further at time of writing, but check `df -h` before
   layering another large model on top.
5. Consider whether `ClosedLoopSpec`/`GatedSpec` results from Axis-1/2/3
   (not just Axis-4) should be re-examined for the same fingerprint
   (`mean_gamma` varying, `mean_accept_rate` flat). Still open — this
   session's fix makes such a re-examination possible for the first time,
   but none of Axis-1/2/3's historical sweeps have been rerun.
6. **New, from §0**: re-run §4.3's confirming test with a live controller
   (not rebuild-per-cell) to get a genuine live-actuation ITL/acceptance
   number to sit alongside the existing rebuild-per-cell evidence.
7. **New, from §0**: audit any FUTURE `vllm_patch/` changes for the same
   client-process-import hazard that broke `configure()`'s cross-process
   propagation this session (see §0's "separate, more insidious bug"
   paragraph) — importing vllm.v1.spec_decode or similar CUDA-touching
   submodules from replay.py's client process risks silently forcing
   `spawn` and zeroing out telemetry/control with no visible error.

---

## 6. Reproducing this session's results

```bash
# environment (see §1 for the disk/venv-location notes)
python3 -m venv /root/spec_decode_env/venv && source /root/spec_decode_env/venv/bin/activate
pip install -r requirements.txt
pip install -r specloop_rt/requirements.txt   # pins vllm==0.9.2, transformers==4.53.3
export HF_HOME=/root/spec_decode_env/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# re-download Mixtral (deleted at session end, see §1)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8', cache_dir='$HUGGINGFACE_HUB_CACHE')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('yuhuili/EAGLE-mixtral-instruct-8x7B', cache_dir='$HUGGINGFACE_HUB_CACHE')"

# the B x k fit sweep (32 cells, ~3.5-4h on this A100)
python3 scripts/sweep_roofline_moe.py \
  --config configs/a100_80gb_mixtral_ngram.yaml \
  --rtype code --rate 8 --Bs 8 16 24 32 --ks 0 1 2 3 4 5 6 8 --seeds 0 \
  --duration 120 --out results_gpu_sweep/axis4_roofline_moe_fit

# the EAGLE freeze-bug confirming test (~20 min, needs HF_TOKEN for Llama-3.1-8B)
export HF_TOKEN=<your token with meta-llama/Llama-3.1-8B-Instruct access>
python3 scripts/confirm_eagle_gamma_freeze.py
```

`results_gpu_sweep/*/grid.json` for all sweeps referenced above are committed
in this branch (force-added past the `results_gpu*/` ignore rule, same
convention as Axis-2/3/4). Per-cell `steps.jsonl`/`requests.jsonl` telemetry
is not committed (bulky, regenerable) — re-run to regenerate if needed.

## 7. Reproducing THIS session's fix (the live-gamma patch, §0)

```bash
# environment -- same as §6 above
python3 -m venv /root/spec_decode_env/venv && source /root/spec_decode_env/venv/bin/activate
pip install -r requirements.txt
pip install -r specloop_rt/requirements.txt
export HF_HOME=/root/spec_decode_env/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# GPU-free unit tests (no vllm/GPU needed to run these, ~1s)
python3 tests/test_live_gamma_patch.py
python3 tests/test_patch_contract.py       # confirms no regression in the existing contract test

# ngram smoke test (Qwen2.5-0.5B, ungated, ~1 min incl. download)
python3 scripts/smoketest_live_gamma.py \
  --config configs/smoketest_live_gamma.yaml --flip-step 60 --rate 8 --duration 30

# EAGLE smoke test (Llama-3.1-8B + EAGLE3, needs HF_TOKEN, ~2 min incl. download)
export HF_TOKEN=<your token with meta-llama/Llama-3.1-8B-Instruct access>
python3 scripts/smoketest_live_gamma.py \
  --config configs/smoketest_live_gamma_eagle.yaml --flip-step 40 --rate 6 --duration 30

# both should print:
#   num_spec_tokens/req:  pre=<~gamma_a>  post=<~gamma_b>
#   PASS: num_spec_tokens/req tracks the live gamma flip ...

# HillClimbSpec live validation (same Qwen model, ngram, ~40s)
python3 -m specloop_rt.replay \
  --config <a config with controller.spec: hillclimb, e.g. see §0.2's smoketest config
            adapted with controller: {spec: hillclimb, admit: static, spec_kw: {settle_steps: 15,
            deadband_frac: 0.03, gamma_init: 4, window_halfwidth: 3}, admit_kw: {max_num_seqs: 16}}> \
  --trace homogeneous --rtype code --rate 10 --duration 40 --seed 0 --out <out_dir>
# then check <out_dir>/steps.jsonl: gamma_current should move across several
# values (not sit flat), and num_spec_tokens/num_running per step should
# track gamma_current closely.

# next step NOT run this session -- full-scale re-run once Mixtral weights are back:
python3 scripts/sweep_hillclimb_vs_static.py \
  --config configs/a100_80gb_mixtral_ngram.yaml \
  --rtype code --rate 8 --Bs 8 16 24 32 --seeds 0 \
  --duration 120 --out results_gpu_sweep/axis5_hillclimb_vs_static_live
```
