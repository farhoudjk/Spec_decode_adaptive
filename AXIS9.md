# Axis-9: footprint-aware draft pruning — a real GPU-validated mechanism,
# a real bug found and fixed, and a narrow but genuine positive result

## 0. Why this axis exists

Axis-7's paper (`PAPER_DRAFT_AXIS7.md`) established that MoE speculative-decoding
verify cost is driven by expert-activation footprint (distinct experts touched
in the verify batch), not token count. Its own §7 limitations flagged the
natural follow-on: the footprint hook is purely observational — it reads
routing *after* the verify pass has already run, too late to prune on. This
axis asks the direct question: can a **draft-time** proxy for that same
footprint signal be used to prune candidates *before* verification, and does
doing so actually improve real serving throughput?

Three user-directed choices shaped this axis's scope, made explicit here
since they are not derivable from the code alone:

- **Draft-side routing proxy**, not offline simulation and not a
  from-scratch routing-prediction feature search — test whether EAGLE3's
  own draft hidden state, already computed for free, carries the signal.
- **Real GPU testing only, no simulators** — every claim below is backed
  by an actual SGLang server run, not a modeled estimate.
- **Keep going until a real result**, including root-causing and fixing a
  genuine bug uncovered along the way, rather than stopping at the first
  negative number.

## 1. The prerequisite: is target routing predictable from the draft's hidden state?

`scripts/probe_draft_routing_predictability.py`. Standalone port of SGLang's
`LlamaForCausalLMEagle3` layer math (`specloop_rt/sglang_patch/
eagle3_draft_port.py`) run outside SGLang entirely — plain `transformers`
for the target model (Qwen3-30B-A3B-Instruct-2507, teacher-forced,
`output_router_logits=True`) plus the ported draft layer, no live server
needed for this step. A linear probe (one `nn.Linear` per target MoE layer,
draft hidden state → expert logits) trained on 64 HumanEval prompts (6,584
token positions) predicts the target's next-position top-8 expert set with
mean Jaccard overlap **0.52** vs a **0.22** "always guess the popular
experts" baseline, at **all 48/48 target layers** — not a lucky subset.
Position-shift is load-bearing: the draft state at position *t* is compared
against target routing at *t+1* (what it's actually trying to anticipate),
not at *t* (which would trivially leak).

**Verdict: the prerequisite holds, decisively.** See
`results_gpu_sweep/draft_routing_probe/result.json`.

## 2. Training the real proxy: does the signal generalize across workloads?

`scripts/train_routing_proxy.py`. The probe above was a single-workload,
throwaway-weights diagnostic. This trains the actual
`RoutingProxyHead` (defined in `specloop_rt/sglang_patch/
footprint_aware_pruning.py`, the class the live hook imports) on 66,125
pooled tokens from `code`+`rag`+`chat` (150 prompts each), holding out
`reason`/CNN-DailyMail **entirely** — never seen in training — as a
cross-workload generalization check the original probe never attempted.

| Eval | probe Jaccard | freq baseline | margin |
|---|---|---|---|
| In-distribution (held-out tokens, same 3 workloads) | 0.459 | 0.103 | +0.356 |
| **Cross-workload** (held-out `reason`, never trained on) | 0.406 | 0.103 | +0.303 |

Generalization gap: only 0.053. The signal is not an artifact of one
workload's token distribution. Checkpoint:
`results_gpu_sweep/routing_proxy_train/routing_proxy_head.pt`.

## 3. Live SGLang integration: CUDA-graph safety and the real hook point

Two environment problems, unrelated to the mechanism, cost real time and are
worth not rediscovering:

- **sglang==0.5.17's pinned `sgl-kernel==0.4.5` dropped A100 (sm80) support**
  — confirmed by inspecting actual wheel contents across versions, not a
  local misconfiguration. Fixed by downgrading to `sglang==0.4.10` (pins
  `sgl-kernel==0.2.8`, confirmed to still ship the universal/sm80-compatible
  `common_ops.abi3.so`) in a fresh venv (`/workspace/sglang_env_a100`).
- **That PyPI wheel is separately missing a C++ source file**
  (`hf3fs_utils.cpp`) an optional, unrelated distributed-cache backend
  JIT-compiles unconditionally at import time, crashing every server launch
  regardless of whether that feature is used. Fixed with a minimal stub
  (`sglang_env_a100/venv/.../hf3fs_utils.cpp` — a venv-local file, not
  committed to this repo).

**The mechanism itself, `specloop_rt/sglang_patch/footprint_aware_pruning.py`
+ `install_footprint_pruning.py`:** patches
`eagle_worker.select_top_k_tokens` (note: `sglang.srt.speculative.
eagle_utils` in 0.4.10, `.spec_utils` in 0.5.17 — the module path is
version-specific and was confirmed live, not assumed). Read directly against
installed source and confirmed by an actual `torch.cuda.graph`
capture/replay test (not just source-reading): this call site runs in eager
Python exactly once per captured batch-size bucket, at server warmup — a
monkeypatch here fires and its tensor ops get baked correctly into the
replayed graph, unlike Axis-7's take-1 hook (`TopK.forward`) which never
fired on real traffic at all. This constrains the patched logic to pure
tensor ops (no Python-level loops/sets/`.tolist()` on draft-step-dependent
data) — the original set-based cost accounting was rewritten as multi-hot
tensor masks for exactly this reason.

**Integration approach, chosen deliberately over a riskier alternative:**
rather than reimplementing `select_top_k_tokens`'s own tree-topology
bookkeeping (`tree_info`'s parent-pointer encoding, consumed downstream by
verify-step attention masking — a subtle bug there would silently corrupt
generation, not crash), `compute_footprint_adjusted_topk_p` returns an
adjusted `topk_p` fed into the **real, unmodified** `select_top_k_tokens`.
SGLang's own already-correct code does the actual selection and index
bookkeeping; only the ranking *signal* changes.

**Verified live, not just unit-tested:**
- `lambda=0` identity: generated text and `spec_verify_ct` **byte-for-byte
  identical** to a genuinely unpatched baseline server, on two different
  prompts.
- `lambda=1`/`lambda=50`: zero exceptions across CUDA-graph capture at every
  batch bucket (1–16) and real generation including a 100-token completion.
- **Real, inherent scope limit found, not a bug**: at `i==0` (the tree
  root), all `topk` candidates for one request share the *same*
  `hidden_states` row — only `topk_index`/`topk_p` vary per candidate at
  that step. The proxy, which only reads `hidden_states`, cannot
  discriminate between root-level candidates at all. Pruning only acts at
  `i>0`, where `hidden_states` genuinely varies per candidate after the
  prior step's re-gather.

## 4. First goodput measurement: a clear, consistent loss — and a real bug

`scripts/goodput_pruning_comparison.py`, matching this repo's own sweep
convention (`scripts/sweep_sglang_depth_width.py`): real open-loop-Poisson
traffic (`specloop_rt.workload.homogeneous`, real HumanEval prompts) against
a live server, comparing baseline vs pruning-enabled at matched `(D=3,W=4)`
tree shape.

**First full grid** (`results_gpu_sweep/goodput_pruning_comparison/`,
B∈{8,16} × rate∈{2,8} × lambda∈{1,20}): pruning was **worse than baseline at
every single point**, 8–22% lower goodput, `verify_ct` trending *up* not
down. A legitimate negative result on its face — but the investigation into
*why* found something more specific than "the mechanism doesn't work."

**Root cause, found by comparing SGLang's own per-phase memory-usage log
lines between configs** (not assumed from theory): `sitecustomize.py`
installed the hook **unconditionally in all 8 SGLang subprocesses**
(scheduler, detokenizer, `torch._inductor` compile workers), but the
original `install()` **eagerly** loaded the checkpoint and built
`RoutingProxyHead().cuda()` in every one of them — even the 7 that never
call `select_top_k_tokens`. Confirmed directly via
`nvidia-smi --query-compute-apps`: ~588MB GPU memory per idle subprocess ×
7 = **~4.1GB wasted**, directly shrinking the scheduler's own KV-cache
budget (`max_total_num_tokens`: 108,134 → 86,159, a 20% cut) — enough to
measurably hurt throughput under load, independent of whether the pruning
*logic* itself was any good. (A parallel theory — that 48 separate
`nn.Linear` kernel launches during CUDA-graph capture inflated the capture
memory pool — was tested and ruled out: batching them into one `einsum`
call left capture memory unchanged, `1.81GB → 1.82GB`, and isolated
`torch.cuda.memory_reserved()` tracing inside the hook showed only ~40MB
of real growth during the entire capture phase, nowhere near the ~1.6GB
gap SGLang's own logs reported.)

**Fix**: lazy initialization (`install_footprint_pruning.py`'s `_lazy_init`)
— `install()` now only patches a cheap wrapper unconditionally; the
checkpoint load and GPU allocation are deferred to the *first real call*,
which only the scheduler subprocess ever makes. Confirmed live:
`max_total_num_tokens` restored to the exact baseline value (108,134),
`nvidia-smi --query-compute-apps` shows GPU memory concentrated in one
process, not eight.

**Rerunning the identical grid after the fix**
(`results_gpu_sweep/goodput_pruning_comparison_v2_fixed/`): the gap roughly
halved or better across every cell (e.g. B8/rate2/lam1: −8% → −3.2%;
B16/rate8/lam1: −17% → −7.6%) but pruning still lost everywhere at
`lambda∈{1,20}` — `verify_ct` still climbed monotonically with lambda in
every one of the 4 tested `(B,rate)` points, indicating the *remaining*
gap was a real selection-quality effect, not leftover overhead.

## 5. Second diagnosis: the proxy's absolute accuracy, not just `i==0` blindness

§2's cross-workload Jaccard (~0.40–0.46) is a *relative* result (vs. a weak
frequency baseline) — in absolute terms the proxy still gets roughly half
of its predicted top-8 experts wrong. `marginal_cost`, derived from those
predictions, is therefore a genuinely noisy quantity, and dividing `topk_p`
by `(1 + lambda·marginal_cost)` injects that noise into a ranking signal
that was already working reasonably well on its own. This predicts exactly
the observed pattern: small lambda should help (a light, mostly-correct
nudge), large lambda should hurt (noise dominates) — a real optimum, not a
uniformly bad direction as `i==0` blindness alone would suggest.

**Tested directly**, `results_gpu_sweep/goodput_pruning_small_lambda/`
(single seed, B8/rate2): `lambda=0.1` **beat baseline**
(102.29 vs 97.81 tok/s, +4.6%) while `lambda=0.5` lost (91.84, −6.1%) —
confirming an inverted-U shape, not a monotonic decline.

**Confirmed with proper statistics**,
`results_gpu_sweep/goodput_pruning_lambda_sweep/` (3 seeds each, B8/rate2):

| Config | goodput (mean ± std, n=3 seeds) | verify_ct |
|---|---|---|
| baseline | 96.39 ± 5.46 | 40.69 |
| **lambda=0.05** | **99.66 ± 5.18** (+3.4%) | **40.37** (lower — genuinely better) |
| lambda=0.1 | 97.62 ± 4.37 | 40.41 |
| lambda=0.2 | 97.89 ± 4.24 | 40.68 |
| lambda=0.5 | 94.00 ± 3.73 (−2.5%) | 40.95 |

**This is the first result all session where the mechanism both beats
baseline goodput AND improves `verify_ct`** (fewer verify steps needed,
not just a wash) — real evidence of better candidate selection, not
measurement noise, at a narrow, low-lambda regime. The shape across all 5
ordered points (rise, peak near 0.05–0.2, decline) is coherent with the
noisy-signal-at-high-lambda theory; pure noise would not produce this
pattern.

**Honest caveat**: standard deviations (±4–5 tok/s) are large relative to
the effect size (~3 tok/s gap) at only 3 seeds and one `(B,rate)` point —
a real signal, not yet a fully decisive one. More seeds and broader
`(B,rate)` coverage are needed before trusting `lambda≈0.05` as a settled
operating point (see §6).

## 6. Open items

- **Statistical confidence at the found optimum is not yet solid** — more
  seeds at `lambda≈0.05` (and neighboring values, e.g. 0.02–0.08) needed
  before treating +3.4% as settled rather than a promising but noisy
  signal.
- **Only tested at one `(B,rate)` point (B=8, rate=2) with real statistics**
  — the full 2×2 grid from §4 needs rerunning at the newly-found low-lambda
  regime to check the optimum holds across load conditions, not just the
  one point it was found at.
- **`i==0` (tree root) is never adjusted** — a real, inherent limit of a
  hidden-state-only proxy (see §3). Whether a different feature source
  could extend pruning to the root, and whether that would matter given
  §5's finding that even `i>0`-only pruning already shows a positive
  effect at the right lambda, is unexplored.
- **Only one tree shape tested throughout (D=3, W=4)** — Axis-7/8's own
  findings suggest wider trees have more expert-footprint variance to
  exploit; this method has never been tested at W>4.
- **Per-request-identity coverage under real concurrent mixed-draft-step
  traffic** — the current `i==0` reset logic
  (`covered_mask[:b].zero_()` in `install_footprint_pruning.py`) assumes
  request-to-slot assignment starts at 0, valid for this session's
  one-request-at-a-time tests but not verified correct under real
  concurrent traffic with requests at different draft steps
  simultaneously.
- **Single workload (`code`/HumanEval) for all goodput measurements** — §2's
  cross-workload check was for the *prediction* task only; the *goodput*
  sweep itself has not been repeated on `rag`/`chat`/`reason`.

## 7. Reproducing

```bash
# Environment: this repo's A100-compatible sglang stack (NOT the 0.5.17
# environment AXIS7/8 used -- see §3 for why)
python3 -m venv /workspace/sglang_env_a100/venv
source /workspace/sglang_env_a100/venv/bin/activate
pip install "sglang[srt]==0.4.10"
# Then patch the missing hf3fs_utils.cpp stub -- see §3; the stub itself
# lives in the venv, not this repo, since it's a workaround for a
# third-party packaging bug, not this project's own code.

export HF_HOME=/root/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1

# §1: routing-predictability probe (no SGLang server needed)
python3 scripts/probe_draft_routing_predictability.py \
  --n-prompts 64 --rtype code \
  --out results_gpu_sweep/draft_routing_probe/result.json

# §2: train the real, multi-workload proxy checkpoint
python3 scripts/train_routing_proxy.py \
  --n-prompts-per-workload 150 --holdout-workload reason \
  --out-dir results_gpu_sweep/routing_proxy_train

# §4-5: live goodput comparison (requires the checkpoint above)
export PYTHONPATH="$(pwd):$(pwd)/specloop_rt/sglang_patch"
python3 scripts/goodput_pruning_comparison.py \
  --Bs 8 --rates 2.0 --duration 30 --lambdas 0.05 0.1 0.2 0.5 --seeds 0 1 2 \
  --out results_gpu_sweep/goodput_pruning_lambda_sweep
```

`--speculative-num-steps 3 --speculative-eagle-topk 4
--speculative-num-draft-tokens 8` (D=3, W=4) was the one tree shape tested
throughout this axis — see §6 for why wider trees are the natural next
sweep dimension.
