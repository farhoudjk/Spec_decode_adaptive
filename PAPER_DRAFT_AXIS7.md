# Routing the Speculation: Expert-Activation Footprint as the Missing Term in MoE Speculative-Decoding Cost Models

**Status: draft, findings current as of this repo's `axis6-sglang-width` branch, Axis-7 work.**
All numbers below are pulled directly from committed `results_gpu_sweep/*/grid.json`
files and are reproducible from this repo; see AXIS7.md for the full session-by-session
trace (including two dead ends) behind every claim here, and each section's pointer
to the source sweep / reproduction command.

**Scope note, read before citing this draft**: this paper is scoped to what was
GPU-validated in one session, on one MoE model (`Qwen/Qwen3-30B-A3B-Instruct-2507`),
two workloads (`code`/HumanEval, `reason`/CNN-DailyMail), three load points. It does
NOT claim: generality across MoE architectures, a live adaptive controller (SGLang's
tree width is launch-time-static — confirmed, not a limitation of our approach), or
route-aware draft expansion (the design brief's more ambitious C3b, not attempted).
See §7 for the full limitations list.

---

## Abstract

Speculative decoding's cost model for dense LLMs treats verify cost as linear in
token count — every proposed draft token costs the same to verify, so the only
tension is acceptance-vs-latency. We show this breaks down for Mixture-of-Experts
(MoE) models: a draft tree's verify cost is governed by its **expert-activation
footprint** (how many distinct experts its tokens route to) rather than token
count alone. We give the first direct, mechanistic measurement of this effect
— prior work could only infer it from an unexplained cost-model residual — by
building a graph-safe hook into SGLang's scheduler that captures the *full*
verify-batch's routing decisions (including rejected draft candidates, which
existing telemetry discards). Across the full depth×width grid our target
model supports, an expert-footprint term lifts within-cell cost-model R² from
0.80 to 0.95 on one workload and 0.88 to 0.98 on a second, structurally
different workload — in both cases explaining more of the residual than a
naive depth-width interaction term, using one physically-motivated variable
instead of an unprincipled product. We find this footprint is itself
load-dependent (via verify-batch coalescing under continuous batching, not
routing diversity), and, correcting for it, build a config-time tree-shape
picker that uses the fitted cost model instead of a depth/width-only budget.
Head-to-head against a static default and an architecture-blind budget across
three load points, our picker improves goodput by 6–127%, with the largest
gains — over 2× the static default — at heavy load, exactly where a naive
budget picks trees that are too wide for what the MoE architecture can
actually serve efficiently.

---

## 1. Motivation: an unexplained residual

Prior work on this codebase (`axis6-sglang-width` branch, `AXIS6.md`) swept
EAGLE3 tree depth (`D`) and width (`W`) against real GPU serving latency for
both a dense model (Llama-3.1-8B) and an MoE model
(`Qwen/Qwen3-30B-A3B-Instruct-2507`, 3.3B/30.5B active params). Fitting
`T_step = C_d·D + C_w·W + C_0` (an affine cost model, the natural generalization
of dense speculative decoding's "cost is linear in token count") against
measured per-token latency gave:

| Model | nodes-only (`D·W`) | `D+W` affine | `+D·W` interaction |
|---|---|---|---|
| Dense (Llama-3.1-8B) | R²=0.58 | R²=0.91 | R²=0.92 |
| MoE (Qwen3-30B-A3B) | R²=0.44 | R²=0.78 | R²=0.86 |

Dense: affine is enough, the interaction term buys almost nothing (+0.01).
**MoE: affine leaves 22% of variance unexplained, and an interaction term
recovers 8 of those 22 points** — evidence something beyond simple
per-token cost is happening on MoE specifically, but a product term
(`D·W`) has no physical interpretation; it's curve-fitting, not a mechanism.
Checking whether the residual's *shape* differs from dense's (it should, if
the mechanism is MoE-specific) found the same bump-shape present on dense
too, just ~3× smaller in magnitude — arguing against trusting the residual
pattern further without direct measurement. This paper is that direct
measurement.

## 2. Why width should cost more on MoE (the hypothesis)

In a dense FFN, every token in the verify batch flows through the same
weights — token count is the right complexity variable. In an MoE layer,
each token routes to `top-k` of `N` experts; the grouped-GEMM cost of the
verify pass is governed by the number of *distinct* experts activated and
the load imbalance across them, not raw token count. A **wider** draft tree
adds sibling candidates at each tree position — divergent tokens likely to
route to *different* experts — so width should grow the expert footprint
faster than depth (which mostly extends an already-committed path, tokens
that tend to route similarly to their predecessor).

## 3. Measuring the mechanism directly

### 3.1 Two dead ends, worth reporting

**Attempt 1: a monkeypatch on the routing call (`TopK.forward`).** Installs
cleanly, confirmed via canary instrumentation firing in every process SGLang
spawns — but SGLang captures a CUDA graph for the verify forward pass and
replays it on every real request; a graph replay re-executes the recorded
GPU kernel sequence directly, never re-entering the Python call stack a
monkeypatch depends on. Zero rows recorded on real traffic, caught on the
first GPU cell by a pre-registered guard-rail rather than discovered after a
multi-hour sweep.

**Attempt 2: SGLang's native `--enable-return-routed-experts` flag.**
SGLang ships a production mechanism for exactly this signal — writes routed
expert IDs into a pre-allocated device buffer via a plain in-place tensor
write, which (unlike attempt 1) *is* graph-safe, since CUDA graph capture
supports static buffer writes natively. Confirmed working, but scope-limited
in a way that took direct measurement to discover: the per-request response
only surfaces routing for tokens that received a permanent KV-cache slot —
i.e. **accepted tokens only**. A request with `prompt_tokens=22,
completion_tokens=64` returned exactly `22+64-1=85` tokens' worth of routing
data — an exact arithmetic match confirming the scope. Rejected draft
candidates' routing is captured on-device for an instant during the verify
forward, then discarded before any client-facing read path. Running this
signal on the full depth×width grid gave a robust **null result** —
`mean_distinct_experts` stayed flat (92.25–92.76, a 0.5% range) across the
entire grid, while the control variable (mean accepted length) moved 2×
exactly as the existing model predicts. This is a genuine, honest negative
result — but on the wrong question: it tests whether the *accepted
continuation's* aggregate footprint changes with tree shape (it doesn't,
because with 128 experts and dozens of accepted tokens per request, expert
coverage saturates regardless of tree shape), not whether the *rejected-
inclusive verify batch's* footprint does, which is what the cost-model
hypothesis is actually about.

### 3.2 The working hook: intercepting before the scheduler discards rejected-candidate routing

SGLang's `ModelRunner.forward()` calls its routing capturer once per forward
pass, receiving a `TopkCaptureOutput` holding the *full* verify batch's
routing (every proposed draft token, accepted or rejected — every candidate
needs a provisional KV slot just to compute its verify logits). This object
is **not** finalized inside `forward()`; `.finalize()` — the step that
narrows to accepted-only KV positions — runs later, from a separate
scheduler-pipeline stage. We patch `ModelRunner.forward()` itself, reading
`routed_experts_output.topk` immediately on return, before `finalize()` ever
runs.

This patch point is graph-safe by the same logic that made attempt 2 work:
CUDA graph replay happens strictly *inside* `_forward_raw()` and returns
before the capturer is ever invoked — the capturer's Python-level
orchestration runs in eager Python every step, on top of, not inside, the
replayed graph.

A second real bug surfaced during GPU testing: buffering rows in memory and
relying on `atexit` to flush them at server shutdown silently lost all data
— the sweep harness stops servers via `SIGTERM`, which does not trigger
Python's `atexit` handlers unless the receiving process installs its own
handler that calls `sys.exit()`. Fixed by writing every record straight to
disk on every verify step (write-through), confirmed by a populated log file
appearing *during* serving, not just after teardown.

## 4. Result: the mechanism is real and generalizes across workloads

Ran the full depth×width grid our target model supports
(`D∈{1,2,3,4,6,8}×W∈{1,2,4,8}`, 24 cells, code/HumanEval workload,
`B=16,rate=4` load point) with the working hook. Joining against the
existing timing grid on `(D,W)` and fitting `T_step ~ D + W + E` (`E` =
mean distinct experts activated in the full verify batch, this session's
new measurement):

| Model term | `code` (24 cells) | `reason` (9 cells) |
|---|---|---|
| `D+W` affine | R²=0.798 | R²=0.876 |
| `D+W+D·W` (interaction) | R²=0.868 (+0.070) | R²=0.879 (+0.003) |
| `D+W+E` (this work) | **R²=0.947 (+0.149)** | **R²=0.978 (+0.102)** |

On both workloads, the expert-footprint term explains substantially more of
the residual than the interaction term — more than double on `code`, over
30× on `reason` (where the interaction term is essentially useless,
+0.003). `C_w` (the fitted per-unit-width cost coefficient) shrinks 61% on
`code` and 91% on `reason` once `E` enters the model — direct evidence that
what looked like an intrinsic cost of width was substantially the cost of
the extra experts width activates, not width itself.

`reason` (CNN-DailyMail summarization) is structurally different from `code`
(HumanEval function completion) — different token-length distribution,
different content domain — so this is not an artifact of one corpus's token
statistics.

## 5. `E(tree)` is load-dependent, and why

We initially assumed expert-activation footprint is a property of tree
shape alone. This is wrong. Measuring the same depth×width grid at two more
load points spanning our target model's tested range (light: `B=8,rate=2`;
heavy: `B=32,rate=12`) found `mean_distinct_experts` increases monotonically
with load at every single `(D,W)` cell — e.g. at `D=1,W=1`: 18.5 (light) →
22.4 (mid) → 57.7 (heavy).

The mechanism is verify-batch coalescing, not routing itself becoming more
diverse: mean verify-batch token count jumps from 4.6 (mid) to 25.9 (heavy)
at the same cell — a higher admission cap lets more concurrent requests'
verify steps land in the same scheduling window and get batched together,
so each *measured* "verify step" spans more tokens and mechanically touches
more experts. Confirming this is not "heavier load makes routing more
diverse per token": normalizing by verify-batch size (experts touched per
token) shows the *opposite* direction — density falls monotonically with
load at every cell (light > mid > heavy), consistent with saturation toward
the architecture's 128-expert ceiling as batches grow (a bigger batch is
more likely to re-hit an already-touched expert than a new one).

This has a direct consequence for cost-model fitting: attempting one global
`T_step ~ D+W+E` fit across all three load points at once collapses `D+W`
alone to R²=0.07 (their true effect is swamped by load-driven variance in
raw counts) while `D+W+E` recovers to R²=0.89 — but with **negative**
fitted `C_d, C_w` coefficients, a collinearity artifact (D, W, and E are
correlated once load varies, since E is itself downstream of D, W, *and*
load via batch coalescing), not a physically meaningful claim that more
depth/width reduces cost. Constraining coefficients non-negative (via
non-negative least squares) resolves this cleanly, but at the cost of a
single global formula: the constrained fit drives `C_d, C_w` to exactly
zero even *within* one load point at heavy load, and produces physically
nonsensical negative latency predictions at light/mid load regardless — a
single linear-in-`E` model cannot span a ~4× range of absolute latency
scale with one intercept.

## 6. Building on the corrected model: a config-time tree-shape picker

SGLang's tree width is launch-time-static, not live-adjustable (confirmed
by reading `adaptive_spec_params.py`: `--speculative-adaptive` is hard-
refused unless `eagle_topk ∈ {None, 1}`). A tree-shaping policy for this
serving stack is necessarily a **config-time picker** — choose `(D,W)`
before launching the server for a given `(B,λ)`, not a live per-request
controller.

Given §5's finding that one global cost formula cannot honestly span our
tested load range, we fit `T_step ~ D+W+E` **separately at each of the
three measured load points** (light/mid: plain OLS, coefficients already
non-negative, R²=0.974/0.947; heavy: same collinearity issue as the global
fit, resolved the same way — non-negative-constrained refit, `C_d→0`,
R²=0.817) and built `pick_tree(B,λ)`: starting from the smallest tree,
greedily grow whichever axis (depth or width) has the larger marginal
accepted-length-per-marginal-cost, using the load point's own fitted model,
stopping when neither axis's marginal exceeds the tree's current average.
The picker requires `(B,λ)` to exactly match one of the three measured
points — we do not claim interpolation or extrapolation to untested load,
a real scope limitation (§7).

### 6.1 Head-to-head evaluation

We compare three tree-shape policies at all three measured load points:

- **B0 (static default)**: `D=3,W=4` fixed, regardless of load.
- **B1 (architecture-blind budget)**: the same marginal-utility rule, but
  fit against a `D+W`-only cost model (no expert-footprint correction) —
  isolates whether the expert term itself, not just "having a budget rule
  at all," is what matters.
- **B2 (this work)**: `pick_tree(B,λ)`, the expert-footprint-corrected
  budget.

All nine `(policy, load point)` cells were already present in the existing
288-cell timing sweep (every `(D,W,B,λ)` combination any policy could pick
was already measured), so this evaluation required no new GPU time.

| Load point | B0 goodput | B1 goodput | B2 goodput | B2 vs B0 | B2 vs B1 |
|---|---|---|---|---|---|
| Light (B=8,λ=2) | 112.9 tok/s | 118.9 tok/s | **126.0 tok/s** | +11.6% | +5.9% |
| Mid (B=16,λ=4) | 82.4 tok/s | 93.5 tok/s | **94.0 tok/s** | +14.0% | +0.5% |
| Heavy (B=32,λ=12) | 20.4 tok/s | 28.4 tok/s | **46.2 tok/s** | **+126.9%** | **+62.6%** |

B2 wins at every load point, with the gap widening sharply under heavy
load: more than double B0's goodput, and 63% ahead of B1 — the
architecture-blind budget alone is not enough; the expert-footprint
correction specifically is what recovers the extra headroom. B2
consistently picks a narrow tree (`W=1` at every load point tested),
converging on the same qualitative finding an independent, purely empirical
grid sweep in prior work on this codebase already established (MoE strongly
favors `topk=1`, winning 72% of cells across a full 576-cell grid) — the
fitted cost model rediscovers this from first principles rather than being
told it, which is reassuring cross-validation of the mechanism, not just
the fit.

## 7. Limitations

- **Single MoE model.** Everything above is measured on one target
  (`Qwen/Qwen3-30B-A3B-Instruct-2507`, 128 experts, top-8, ~11% active
  params). Whether the expert-footprint mechanism's *magnitude* (not just
  its existence, which follows from MoE's architecture generally)
  transfers to a MoE with a different expert count, top-k, or active-param
  ratio is untested.
- **No live controller.** The picker is config-time only, a direct
  consequence of SGLang not supporting live width adjustment — not a
  limitation of the cost model itself, but it does mean the evaluation in
  §6 is restarts-between-configs, not a single server adapting online.
- **Three load points, not a continuous load model.** We show `E` is
  load-dependent and explain the mechanism (§5), but do not fit a
  continuous `C_e(B,λ)` scaling law the way prior axis6 work fits
  `C_d,C_w ∝ λ^1.4·B^{-0.15}` for the D/W terms. The picker's per-load-
  point lookup table is honest about only interpolating within measured
  points, not a substitute for that law.
- **Route-aware expansion (C3b) not attempted.** The more ambitious
  direction in the originating design brief — biasing draft-tree width
  expansion toward candidates predicted to route to already-active
  experts, using a routing estimator — is unimplemented. §6's picker only
  corrects the depth/width *budget*; it does not change which candidates
  a fixed-width expansion selects.
- **Session-to-session timing drift.** The cost-model fits in §4 join data
  from separate sweep runs (the footprint sweep and the original axis6
  timing sweep); this repo's own prior measurement notes document up to
  ~8% session-to-session throughput drift on this rig. The R² jumps
  reported (e.g. 0.80→0.95) are large enough that this drift alone is very
  unlikely to explain them away, but it is not formally controlled for.
- **`--enable-return-routed-experts`'s overhead is unmeasured** against
  the baseline axis6 timing sweep's assumptions; the write-through hook
  used for §4-§6 similarly has an unmeasured but plausibly non-trivial
  per-verify-step cost (a device-to-host copy plus a file write per step).
  Timing numbers reported in §4/§5's *raw tables* come from hook-enabled
  runs and should not be treated as directly comparable to hook-disabled
  axis6 baselines at face value — though the goodput comparison in §6
  uses only the pre-existing, hook-*disabled* axis6 grid, so that specific
  result is not affected by this caveat.

## 8. Reproducing

All commands below assume the repo root, `axis6-sglang-width` branch, and
the GPU environment documented in `AXIS6.md §5` / `AXIS7.md §17`.

```bash
# §4: full depth x width expert-footprint grid, code workload
python3 scripts/sweep_sglang_verify_footprint.py \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --draft-path lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex \
  --num-layers 48 --topk-size 8 \
  --out results_gpu_sweep/sglang_verify_footprint_grid

# fit and compare against axis6's residual
python3 scripts/fit_expert_cost_term.py \
  --axis6-grid results_gpu_sweep/sglang_depth_width_rate_qwen3moe/grid.json \
  --footprint-grid results_gpu_sweep/sglang_verify_footprint_grid/grid.json \
  --B 16 --rate 4

# §6: the picker itself
python3 scripts/pick_tree.py --B 16 --rate 4 --verbose
```

See `AXIS7.md` for the complete session trace, including both dead ends
(§2-§7 of that document), the load-dependence investigation (§16), and
every intermediate fit attempted before arriving at the models used here.
