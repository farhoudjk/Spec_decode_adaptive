# Axis-5: MoE roofline validation, and a proposer-freeze bug that reopens Axis-4

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
   Axis-4's "the null is structural" conclusion. See §4 — this is the most
   important finding of the session and needs to be the first thing read
   next session.

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

1. **Re-run `sweep_hillclimb_vs_static.py` correctly** — add
   `cfg["runtime"]["gamma_init"] = k` (for static arms) to `run_cell`,
   matching `sweep_roofline_moe.py`'s pattern. This fixes the static-k
   baselines. It does NOT make `HillClimbSpec` a real live experiment — see
   next point.
2. **Decide how to evaluate "adaptive" given the freeze.** `HillClimbSpec` is
   correctly implemented (batch-gated window from the k\*(B) formula, settle-
   then-compare timing fixed after an earlier bug where it compared ITL every
   4 steps against an EMA with a ~10-step time constant and never converged —
   see the class docstring) but cannot be validated as "adaptive" against the
   real proposer in this vLLM version. Options discussed but not decided:
   (a) redefine "adaptive" as a per-cell-rebuild k\*(B) lookup instead of a
   live controller — testable today with the existing pattern; (b) attempt
   the vLLM patch to make the proposer live; (c) treat the roofline
   characterization + the freeze-bug finding as the deliverable and stop.
3. **EAGLE+Mixtral KV-head mismatch (§3.1)** — not revisited. Would need
   either a different EAGLE-Mixtral checkpoint with matching
   `num_key_value_heads`, or a vLLM patch around the `NotImplementedError`.
4. **Mixtral weights need re-downloading** (deleted for disk space, see §1).
5. Consider whether `ClosedLoopSpec`/`GatedSpec` results from Axis-1/2/3
   (not just Axis-4) should be re-examined for the same fingerprint
   (`mean_gamma` varying, `mean_accept_rate` flat).

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
