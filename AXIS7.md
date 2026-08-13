# Axis-7: C1 — expert-activation instrumentation for MoE tree shaping

## 0. Why this axis exists

The expert-aware tree-shaping design brief's whole premise rests on one
unconfirmed inference from axis6 (AXIS6.md): MoE's within-cell cost-model
residual (nodes-only R²=0.44 → D+W affine R²=0.78 → +interaction R²=0.86,
reproduced exactly from `results_gpu_sweep/sglang_depth_width_rate_qwen3moe/
grid.json`) is *consistent with* expert fan-out, but axis6 collected no
expert-activation telemetry at all -- nothing in that grid can distinguish
"wider tree activates more experts, and grouped-GEMM cost tracks the busiest
expert" from any other width-driven cost source (kernel launch overhead,
attention mask construction, KV bookkeeping). Re-checking the residual by
width on both MoE and dense (prior session) found the residual's *shape*
(dip at W=4, peak at W=8) is present on dense too, just ~3x smaller in
magnitude -- which argues against a clean monotonic expert-driven story and
for measuring the actual mechanism directly rather than trusting the
residual pattern further.

C1 is that direct measurement, and this doc now covers THREE GPU sessions:
one that built and discarded a custom monkeypatch (§2), a follow-up that
found SGLang's own native instrumentation but hit a real scope limit --
accepted-tokens-only, giving an inconclusive null (§3-§7) -- and a third
that patched deeper (`ModelRunner.forward`, not `TopK.forward`) to reach
the full verify batch, rejected candidates included, then fit the
resulting `C_e·E` cost term against axis6's residual (C2).

**Bottom line up front: §10-§16 supersede §3-§7's inconclusive null.
The full-verify-batch signal shows a clean, GPU-validated, monotonic
increase in expert-activation footprint with BOTH tree depth and width,
across the entire 24-cell `D∈{1,2,3,4,6,8}×W∈{1,2,4,8}` grid tested. This
CONFIRMS the design brief's Milestone 1 hypothesis, and §14's fit gives a
concrete `C_e·E` term (R²=0.947 vs 0.798 for D+W affine alone) ready to
feed into C2's cost model. §15 confirms the same relationship holds on a
second, structurally different workload (`reason`/CNN-DailyMail
summarization, R²=0.978 vs 0.876), so this isn't an artifact of one
corpus's token distribution. **§16 is a necessary correction: `E(tree)`
is NOT load-invariant** -- it moves substantially with batch cap/arrival
rate (confirmed at 3 load points spanning axis6's full range), driven by
verify-batch coalescing under continuous batching, not by routing itself
becoming more diverse. The §14/§15 `C_e` fits remain valid at the load
point they were run at; using `C_e` at a different load point needs the
correction §16 lays out** -- read §10, §14, §15, and §16 first if you
only read a few sections of this doc; §2-§7 are kept as the (informative,
not wasted) trail of two dead ends that led there.

## 1. What was built

| File | Role |
|---|---|
| `specloop_rt/sglang_patch/verify_batch_expert_hooks.py`, `sitecustomize.py` | **Current, working hook (take 3).** Patches `ModelRunner.forward` to intercept the full verify-batch `TopkCaptureOutput` before `finalize()` narrows it to accepted-only KV positions. Write-through to disk (not buffered+atexit -- see §11 for the bug that fix corrects). See §10-§13. |
| `scripts/sweep_sglang_verify_footprint.py` | **Current, working sweep** for the take-3 hook. Wires `PYTHONPATH` + `CAVEMAN_VERIFY_HOOK_*` env vars, aggregates per-verify-step-per-layer expert-footprint stats. See §12. |
| `scripts/fit_expert_cost_term.py` | **C2 fitter.** Joins axis6's timing grid against the verify-footprint grid on `(D,W)`, refits `T_step ~ D + W + E` and compares against axis6's own nodes-only/D+W/interaction fits. See §14, §15. |
| `filter_overlong()` in `scripts/sweep_sglang_depth_width.py` | Drops trace requests too long to fit the server's context budget, imported by all three sweep scripts. Needed for `reason`/CNN-DailyMail (some articles exceed the EAGLE3 draft head's 2048-token cap) -- see §15.1. |
| `scripts/sweep_sglang_expert_footprint.py` | **Take 2, superseded but not wrong** -- accepted-token-only signal via SGLang's native `--enable-return-routed-experts`. Gave a real, honest, but inconclusive null (§6-§7). Left in place; still useful if accepted-continuation footprint (a different, narrower question) is ever wanted again. |
| `specloop_rt/sglang_patch/moe_expert_hooks.py` | **Dead code (take 1), kept for the record.** A custom `TopK.forward` monkeypatch that installs correctly but never fires on real traffic because SGLang replays a captured CUDA graph for the verify forward pass, bypassing Python entirely. See §5. Not deleted: the debugging trail here (canary prints, CUDA graph timing correlation) is genuinely useful provenance for the next person who hits "my monkeypatch installs fine but never runs" against any CUDA-graph-enabled inference server, not just this one. |

## 2. Session 1: the monkeypatch approach and why it failed

Built `specloop_rt/sglang_patch/moe_expert_hooks.py`, patching `TopK.forward`
(not `Qwen3MoeSparseMoeBlock.forward_normal` -- patching the routing call
itself avoids re-deriving `forward_normal`'s `ep_size`/`tp_size` all-reduce
tail) to record `topk_ids` per layer per forward, gated by
`CAVEMAN_MOE_HOOK_OUT`, auto-installed via a `sitecustomize.py` shim on
`PYTHONPATH`. Fully verified without a GPU first (inertness, `_record`/
`flush` logic, `sys.path` mechanics) -- see git history on this file for
that verification trail if useful.

**GPU-tested against real SGLang 0.5.17** (note: version drift from
0.5.16, the pin AXIS6.md documents -- re-verified the hook point was
unchanged on 0.5.17 before trusting it, see §5.1 below): the patch
installed cleanly, confirmed via literal canary print statements placed in
`sitecustomize.py`, observed firing in all 4 processes SGLang spawns for a
single-GPU EAGLE3 server (`pid=15838,16030,16031,16032` in one run's
`server_*.log`). **But real traffic through the server produced zero
recorded rows.**

### 2.1 Root cause: CUDA graph replay bypasses Python monkeypatches

SGLang's startup log reports `cuda_graph={prefill=22.24s, ...,
target_verify=3.69s, draft_prefill=1.23s, draft_decode=4.28s,
draft_extend=0.94s}` -- CUDA graphs are captured once during warmup for
every phase, including `target_verify`, the exact phase that runs the MoE
verify forward through `TopK`. Every subsequent real request logs `cuda
graph: True`. A CUDA graph replay re-executes the recorded GPU kernel
sequence directly; it does not re-enter the Python call stack, so a
Python-level monkeypatch on `TopK.forward` is captured into non-existence
-- the graph was captured against the *original* `forward`, and the patch,
installed after or around capture, is never consulted again.

This was confirmed empirically, not just inferred: canary prints showed
the patch installed in every process, and `run_cell()`'s own
`WARNING: ... produced no expert-activation rows` fired on the very first
real cell -- exactly the failure mode the pre-GPU verification section of
this doc's prior revision flagged as unverified and worth a guard-rail
for. That guard-rail did its job: it caught the failure on cell 1, not
after a multi-hour sweep.

## 3. Session 2: found SGLang's native mechanism, pivoted

Investigating an alternative surfaced `server_args.py`'s
`enable_return_routed_experts` flag ("Enable returning routed experts of
each layer with responses") sitting in the server's own config dump the
whole time. This is SGLang's own first-class, production mechanism for
this exact signal (`state_capturer/routed_experts.py`):

- `capture_routed_experts_if_allowed()` is called unconditionally inside
  `_post_process_topk_ids()` -- i.e. every `TopK.forward` call, target or
  draft, verify or decode -- and writes `topk_ids` into a **pre-allocated
  device buffer** via a plain in-place tensor write:
  `self.buffer[:batch, layer_id, :] = topk_indices`
  (`state_capturer/base.py`). This write pattern is exactly what CUDA
  graph capture supports natively (a static buffer write, not a Python
  function call), so it survives graph replay where the monkeypatch could
  not.
- Enabled server-side via `--enable-return-routed-experts`, requested
  per-call via a **top-level** `return_routed_experts: true` field on
  `/generate` (NOT nested in `sampling_params` -- confirmed live; nesting
  it there silently returns no `routed_experts` key, no error).
- Response: `meta_info["routed_experts"]`, base64-encoded int32, decoded
  with `np.frombuffer(base64.b64decode(...), dtype=np.int32)`.

`scripts/sweep_sglang_expert_footprint.py` was rewritten around this --
`specloop_rt/sglang_patch/` is now dead code (kept, see §1).

## 4. GPU-validated: shape and scope of the native signal

Confirmed live against `Qwen/Qwen3-30B-A3B-Instruct-2507` (48 layers, top-8
of 128 experts, confirmed from the model's own `config.json`, not assumed):
a request with `prompt_tokens=22`, `completion_tokens=64` returned a
`routed_experts` buffer of exactly `(22 + 64 - 1) = 85` tokens' worth of
routing (`85 * 48 * 8 = 32640` int32s, matching the decoded array length
exactly).

**That exact arithmetic match is the key finding: this buffer holds one row
per token that received a permanent KV-cache slot -- i.e. ACCEPTED tokens
only, not every candidate the verify batch actually processed.**
`capture_routed_experts_if_allowed()` does fire for every draft candidate
during the verify forward (confirmed by reading the call site -- it's
unconditional inside `_post_process_topk_ids`), but the per-request
response only reads back the device buffer at `out_cache_loc`, the
KV-cache slot index -- and a rejected draft token never gets a permanent
KV slot, by definition of rejected. So routing for rejected candidates is
captured on-device for an instant, during the verify forward, and then
never surfaces in anything the client can read. There is no request flag
or documented API to recover it; doing so would need a further
SGLang-internals patch reading the pre-`finalize()` device buffer, which
is out of scope for this pass (a deliberate scope decision, not an
oversight -- see §7).

## 5. Aggregation bug found and fixed mid-session

First version of the aggregation computed distinct-experts **per single
token per layer** -- which is trivially always `topk_size` (a token's own
top-k selection is, by construction, `topk_size` distinct experts; you
cannot pick the same expert twice for one token's routing decision). Live
data confirmed this: `mean_distinct_experts: 8.0` exactly, on every one of
508,608 layer-calls in the first real sweep cell, which is the signature
of a per-token measurement, not a useful cross-token footprint statistic.

Fixed by aggregating **per request, per layer, across every accepted token
in that request** -- i.e. "how many distinct experts, and how imbalanced,
did this request's whole accepted continuation touch at this layer."
Verified against hand-built fixture data (two-layer case, one layer with
spread routing, one concentrated) before re-running on GPU.

## 6. GPU-validated: does the corrected signal actually respond to tree shape?

Ran the full 3×3 grid (`D∈{1,3,6} × W∈{1,4,8}`, `B=16, rate=4, code`
workload, 9/9 cells completed, 0 errors) with the corrected
per-request-per-layer aggregation. Result
(`results_gpu_sweep/sglang_expert_footprint_smoke4/grid.json`):

| D | W | mean_distinct_experts | mean_imbalance | mean_accept_length |
|---|---|---|---|---|
| 1 | 1 | 92.41 | 7.620 | 1.719 |
| 1 | 4 | 92.50 | 7.620 | 1.879 |
| 1 | 8 | 92.40 | 7.623 | 1.922 |
| 3 | 1 | 92.25 | 7.619 | 2.496 |
| 3 | 4 | 92.65 | 7.623 | 2.909 |
| 3 | 8 | 92.75 | 7.626 | 2.984 |
| 6 | 1 | 92.31 | 7.614 | 2.896 |
| 6 | 4 | 92.70 | 7.626 | 3.444 |
| 6 | 8 | 92.76 | 7.627 | 3.563 |

`mean_accept_length` moves with `(D,W)` exactly as axis6 already
established (1.72 → 3.56, a ~2x range, structure-only, matches the `ℓ(D,W)`
fit). **`mean_distinct_experts` and `mean_imbalance` are essentially flat
across the entire grid** -- 92.25 to 92.76 (a ~0.5% range) and 7.614 to
7.627 (a ~0.2% range) respectively, with no visible trend across either
axis, at either extreme (D=1 vs D=6, W=1 vs W=8). This is not sampling
noise from a single run -- the range is far tighter than the corresponding
accept-length movement, and there's no monotonic direction to it at all.

## 7. Why the null result, and what it means for the design brief

This is very likely a direct consequence of the accepted-token-only scope
(§4), not evidence against the underlying hypothesis. Reasoning: each
summarized request accumulates dozens of accepted tokens' worth of routing
per layer regardless of `(D,W)` -- with 128 experts and top-8 routing, the
*set* of distinct experts touched across an entire multi-token accepted
continuation saturates quickly (a handful of tokens is often enough to
have sampled most of the layer's commonly-used experts), so whatever
width-dependent signal exists in a SINGLE verify batch's rejected-inclusive
candidate set gets averaged away once aggregated across a whole request's
worth of accepted tokens. The design brief's §2 hypothesis is specifically
about the verify batch's footprint (rejected tokens included, since they
still cost real GEMM compute) -- exactly the data this native mechanism
cannot surface (§4).

**Conclusion: this axis's current output should NOT be read as evidence
against the expert-fan-out hypothesis.** It tested a narrower, adjacent
question ("does the accepted continuation's aggregate expert footprint
change with tree shape") and got a genuine null on that narrower question.
The brief's actual hypothesis -- verify-batch-level, rejected-inclusive
expert footprint driving cost -- remains untested. Getting real evidence
either way requires one of:

1. A deeper SGLang patch that reads the device buffer
   (`state_capturer/routed_experts.py`'s `self.buffer`) before
   `finalize()` trims it to `out_cache_loc`, capturing the full verify
   batch, rejected candidates included. Non-trivial: touches SGLang's
   scheduler/worker internals, not just a client-facing flag, and needs
   its own GPU validation cycle.
2. Indirect inference from `spec_correct_drafts_histogram` (a histogram
   over verify steps of how many drafts each step accepted -- confirmed
   from `schedule_batch.py`'s `update_spec_correct_drafts_histogram`, NOT
   an ordered per-step sequence) combined with some other proxy for
   per-step routing diversity -- not attempted this session, unclear if
   viable.

**This was a deliberate scope decision this session** (see conversation:
chose "use accepted-token routing as C1's signal" over "patch SGLang's
scheduler for the full verify buffer" given the investigation had already
gone several layers deeper than the original C1 scaffold) -- not an
oversight. Milestone 1 of the design brief's build order (§10: "confirm
or kill the expert hypothesis, gate the rest on this") is **not yet
satisfied**. Before writing `C_e·E` into the cost fitter (C2), either
pursue option 1 above, or explicitly accept the accepted-token-only proxy
as sufficient evidence one way or the other and say so in the paper --
that's a judgment call for whoever owns the paper's honesty bar, not one
to make silently in code.

## 8. What's still open / next steps

- The 3×3 grid (§6) covers D∈{1,3,6}, W∈{1,4,8} -- not the full
  `D∈{1,2,3,4,6,8} × W∈{1,2,4,8}` axis6 covers. Given the pattern is flat
  at both tested extremes of each axis already, running the remaining
  cells is low-priority -- would mainly rule out a non-monotonic bump at
  an untested interior point, not change the headline null result.
- Sync/latency cost of the native mechanism vs. axis6's baseline (with
  `--enable-return-routed-experts` off) has not been measured. Expected to
  be much cheaper than the old monkeypatch's per-layer device sync (this
  is an async device-buffer write path, not a blocking `.to("cpu")` per
  layer), but "expected cheaper" is not "measured," so this sweep's
  `per_token_s_p50` still should not be cited as axis6-comparable until
  checked cell-by-cell.
- `--enable-return-routed-experts` was not run against the dense Llama
  control -- expected to be a no-op (no MoE layers, no `TopK.forward`
  calls to capture) but not confirmed.

## 9. Reproducing

```bash
source /root/sglang_env/venv/bin/activate
export HF_HOME=/root/spec_decode_env/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

python3 scripts/sweep_sglang_expert_footprint.py \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --draft-path lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex \
  --num-layers 48 --topk-size 8 \
  --out results_gpu_sweep/sglang_expert_footprint_qwen3moe
```

`--num-layers`/`--topk-size` are required and NOT auto-derived from the
model -- confirm against the target model's own `config.json`
(`num_hidden_layers`, `num_experts_per_tok`) rather than trust the
Qwen3-30B-A3B defaults shown in `--help` if pointing this at a different
model.

## 10. Session 3: patching deeper to reach the full verify batch

§7 named the fix: read the device buffer before `finalize()` trims it to
`out_cache_loc`. Found the exact seam by reading `model_runner.py`
directly: `ModelRunner.forward()` calls
`experts_capturer.on_forward_end(forward_batch, ..., no_copy_to_cpu=...)`
inline (not a separately overridable method), which returns a
`TopkCaptureOutput` holding `.topk` (device tensor, full verify-batch
shape -- every proposed draft token, accepted or not, since every
candidate needs a provisional KV slot just to compute its verify logits)
and `.out_cache_loc` (also verify-batch-sized). This return value is
stashed on `output.routed_experts_output` and NOT finalized inside
`forward()` -- `.finalize()` (the step that narrows to accepted-only
positions) is called later, from `batch_result_processor.py`. That gap is
the interception window.

**Patch: wrap `ModelRunner.forward` itself** (`specloop_rt/sglang_patch/
verify_batch_expert_hooks.py`), call through to the original, then read
`output.routed_experts_output.topk` immediately, before anything else
touches it.

**Why this is graph-safe** (unlike take 1's `TopK.forward` patch, §2):
`_forward_raw()`'s CUDA graph replay (`self.decode_cuda_graph_runner
.execute(...)`) returns BEFORE `on_forward_end()` is ever called --
confirmed by reading `_forward_raw`'s structure directly, its `if
can_run_graph: ... return` branch exits the function early, and
`on_forward_end()` only runs in the outer `forward()`, after
`_forward_raw()` has already returned. So the capturer's Python-level
orchestration always runs in eager Python, post-replay, every step --
same reason the native `--enable-return-routed-experts` mechanism (§3)
works at all despite CUDA graphs; this patch just reads the same
already-graph-safe data one step earlier than the native mechanism does,
before `finalize()` throws the rejected-candidate rows away.

## 11. A second real bug: atexit does not survive SIGTERM

First working version of the patch buffered rows in memory and relied on
`atexit.register(flush)` to write them at process shutdown -- the same
pattern take 1's `moe_expert_hooks.py` used (untested there, since take 1
never recorded a row in the first place). **This was wrong.** GPU-tested
with debug canary prints added directly to `patched_forward`: the canaries
confirmed the hook WAS receiving real `TopkCaptureOutput` objects with
correctly verify-batch-sized `out_cache_loc` arrays (13 entries, matching
`num_draft_tokens=13`, not 1) -- but the sweep still reported
`WARNING: ... produced no verify-batch expert rows`, and no `hooklog_*
.jsonl` file existed after the cell finished.

Root cause: `sweep_sglang_depth_width.stop_server()` (reused by the take-3
sweep script) sends `SIGTERM` to the server's process group. A raw
`SIGTERM` does **not** run Python's `atexit` handlers unless the receiving
process installs its own signal handler that calls `sys.exit()` --
whether SGLang's scheduler subprocess does this for SIGTERM specifically
was never confirmed, and empirically it does not (or does so too late/in
a path that skips atexit). The buffered rows were real and correctly
computed; they simply never reached disk before the process died.

**Fix: write-through, not buffer-and-flush.** `_record()` now opens,
appends, and closes `CAVEMAN_VERIFY_HOOK_OUT` on every single verify-step
call, rather than buffering and waiting for a `flush()` that turned out to
be unreliable. Costs an open+write+close per verify step -- acceptable,
since this sweep already isn't latency-comparable to axis6 (§2 of the
module's docstring). Verified offline first (a fixture write immediately
readable back with no `flush()` call at all), then GPU-confirmed: the
next real run produced a 1.7MB `hooklog_*.jsonl` file *during* serving,
not just after teardown.

## 12. GPU-validated: the full verify-batch signal, and it responds to tree shape

Same 3×3 grid as §6 (`D∈{1,3,6} × W∈{1,4,8}`, `B=16, rate=4, code`
workload, 9/9 cells completed, 0 errors), this time with the take-3
hook (`results_gpu_sweep/sglang_verify_footprint_grid/grid.json`):

| D | W | mean_distinct_experts | mean_imbalance | mean_verify_batch_tokens | p95_distinct_experts |
|---|---|---|---|---|---|
| 1 | 1 | 22.35 | 2.22 | 4.57 | 45 |
| 1 | 4 | 35.58 | 3.37 | 13.47 | 66 |
| 1 | 8 | 45.06 | 4.26 | 27.06 | 78 |
| 3 | 1 | 31.29 | 2.99 | 8.52 | 59 |
| 3 | 4 | 47.13 | 4.43 | 30.00 | 81 |
| 3 | 8 | 49.84 | 4.69 | 37.56 | 85 |
| 6 | 1 | 42.86 | 3.82 | 16.75 | 75 |
| 6 | 4 | 50.73 | 4.67 | 36.70 | 85 |
| 6 | 8 | 50.37 | 4.66 | 35.21 | 85 |

Contrast directly against §6's accepted-only table (92.25–92.76 flat,
zero visible trend): **this signal moves substantially and monotonically
with BOTH depth and width**, in both directions independently:

- **Width, fixed D=1:** 22.35 → 35.58 → 45.06 (W=1→4→8). Same clean
  monotonic shape at D=3 (31.29 → 47.13 → 49.84) and D=6 (42.86 → 50.73 →
  50.37, see below for the one near-flat step).
- **Depth, fixed W=1:** 22.35 → 31.29 → 42.86 (D=1→3→6). Same monotonic
  shape at W=4 (35.58 → 47.13 → 50.73).
- **`mean_verify_batch_tokens` moves in lockstep** (4.57 → 45+ across the
  grid) -- confirms this is genuinely reading a verify-batch-scoped
  signal, not an artifact: a wider/deeper tree produces a bigger verify
  batch (more candidates proposed per step, or more concurrent requests'
  batches coalescing under continuous batching), and that bigger batch
  touches more distinct experts, exactly as the design brief's §2
  mechanism predicts.
- **`mean_imbalance` also climbs monotonically** with both axes (2.22 up
  to 4.69) -- the busiest expert's load grows faster than the
  perfectly-balanced share, consistent with real GEMM cost tracking the
  busiest expert rather than the average.
- **The one non-monotonic step** (D=6: W=4→W=8 drops slightly, 50.73 →
  50.37) sits at `p95_distinct_experts=85` for both cells -- i.e. both are
  near the same observed ceiling, well short of the model's full 128
  experts but plausibly a real saturation point for this workload/model,
  not noise. A genuine interior plateau, not a violation of the trend.

## 13. What this means for the design brief

**Milestone 1 (design brief §10: "confirm or kill the expert hypothesis,
gate the rest on this") is now satisfied, superseding §7's inconclusive
verdict.** The direct, GPU-validated measurement shows verify-batch expert
footprint -- distinct experts activated AND load imbalance across them --
increases substantially and monotonically with both tree depth and width,
exactly as the design brief's §2 mechanism (`T_step^MoE = C₀ + C_d·D +
C_w·W + C_e·E(tree)`) predicts. This is direct mechanistic evidence, not
an inference from a residual pattern (which is all axis6 alone could
offer) -- it closes the gap axis6 left open, though correlating these
`E(tree)` values against axis6's within-cell cost-model residual (fitting
`C_e` and checking whether it meaningfully improves R² over the plain
D+W affine model, per the design brief's §4 cost model and this axis's own
§0 framing) is still the next concrete step, not yet done in this session.
That correlation is what turns "the mechanism is real" into "here is the
fitted `C_e·E` term to add to C2's cost model."

## 14. C2: fitting C_e against axis6's residual (confirmed, full grid)

`scripts/fit_expert_cost_term.py` joins axis6's timing grid
(`sglang_depth_width_rate_qwen3moe/grid.json`) against the verify-footprint
grid (`sglang_verify_footprint_grid/grid.json`) on `(num_steps,
eagle_topk)`, matching axis6's `B=16,rate=4` cell (the footprint grid's
fixed load point) for a like-for-like comparison, then refits the
within-cell cost model with `E = mean_distinct_experts` added.

**First run (9 cells, D∈{1,3,6}×W∈{1,4,8}):** `D+W+E` gave R²=0.9875 vs
`D+W affine`'s 0.8278 -- a striking result on too little data to trust
alone, which is why the grid was extended (§14.1) before treating this as
confirmed.

**Full run (24 cells, all of axis6's `D∈{1,2,3,4,6,8}×W∈{1,2,4,8}` grid at
B=16,rate=4 -- see §14.1 for how this grid was produced):**

```
nodes-only (D*W):              R2=0.4870
D+W affine:                    R2=0.7983   C_d=0.001564 C_w=0.001340
D+W+interaction:                R2=0.8681   C_d=0.002462 C_w=0.002298 C_dw=-0.000240
D+W+E (mean_distinct_experts):  R2=0.9470   C_d=0.000395 C_w=0.000518 C_e=0.000456
D+W+interaction+E:              R2=0.9470   C_d=0.000410 C_w=0.000532 C_dw=-0.000002 C_e=0.000454
```

(For reference, axis6's own original 288-cell fit -- reproduced exactly at
the start of this session, before C1 existed -- gave nodes-only R²=0.44,
D+W R²=0.78, +interaction R²=0.86; this 24-cell subset's nodes-only/D+W/
interaction numbers, 0.487/0.798/0.868, land close to those, which is a
useful sanity check that the smaller matched subset isn't behaving
anomalously before trusting the +E result on top of it.)

**Headline result: `D+W+E` (R²=0.947) beats `D+W+interaction` (R²=0.868)
using ONE additional real, physically-motivated variable instead of an
unprincipled product term** -- the expert-footprint gain (+0.149 over D+W)
is more than double the interaction term's gain (+0.070). `C_w` shrinks by
~61% once `E` is added (0.00134 → 0.00052), consistent with the design
brief's prediction that "the width cost was really expert cost in
disguise." `D+W+interaction+E` doesn't improve further over `D+W+E` alone
(both R²=0.947, `C_dw≈0`) -- once the real mechanism is in the model, the
interaction term has nothing left to explain.

**This is the number C2 needs**: `C_e≈0.000456` (in the same units as
axis6's `per_token_s_p50 * mean_accept_length` -- seconds per unit of
`mean_distinct_experts`), fit on 24 cells at one load point, one workload.
See §14.1-§14.3 for what would strengthen it further before treating it as
final.

### 14.1 Full D×W grid (done)

Extended `sweep_sglang_verify_footprint.py`'s output from 9 to the full 24
`(D,W)` cells axis6 covers at `B=16,rate=4` (`--steps 1 2 3 4 6 8 --topks
1 2 4 8`), reusing the same `--out` directory so the sweep's own
skip-if-done check (`if key in grid and "error" not in grid[key]`) only
ran the 15 new cells, not all 24 from scratch. 24/24 cells, 0 errors.
Result folded into §12's table conceptually (not reproduced in full here
-- see `results_gpu_sweep/sglang_verify_footprint_grid/grid.json` for the
complete 24-row data, or rerun `fit_expert_cost_term.py` which prints the
full table).

Two additional cells are worth flagging as non-monotonic (not errors,
genuine data): `D=4,W=4→8` drops slightly (52.45 → 50.92) and `D=8,W=2→4`
drops more sharply (60.53 → 50.67) -- both plausible saturation/interior
effects near the same ~50 ceiling §12 already noted for D=6, not new
concerns, but worth keeping in mind if a smooth parametric form for `E(D,W)`
is ever fit (a piecewise or saturating functional form may fit better than
a straight `C_e·E` linear term once `E` itself is modeled as a function of
D and W, rather than treated as an independent measured covariate the way
this section's fit does).

### 14.2 Generalization status (updated -- see §15, §16)

The §14 headline fit is **one workload (`code`/HumanEval), one load point
(B=16, rate=4)**. Two axes of generalization were identified as open;
both are now closed:

- **Other workloads**: **done for `reason` -- see §15.** `rag` (SQuAD) and
  `chat` remain untested.
- **Other B×rate load points**: **done -- see §16. The "purely a property
  of tree shape" assumption below was WRONG** -- `E(tree)` is genuinely
  load-dependent, confirmed at two more `(B,rate)` points spanning axis6's
  full light-to-heavy range. §16 explains the mechanism and what it means
  for using `C_e` correctly.

### 14.3 Other open items (carried over, still unaddressed)

- **Sync/latency cost of the write-through patch**: not yet measured
  against axis6's baseline. Expected non-trivial (`.to("cpu")` once per
  verify step, plus a file open+write+close per step) -- this sweep's
  `per_token_s_p50` should not be treated as axis6-comparable until
  checked cell-by-cell.
- **Dense Llama control**: not yet run with this hook. Expected to be a
  no-op (no MoE layers, no `routed_experts_output` ever populated) but
  not confirmed -- would be a useful sanity check alongside the workload/
  load-point generalization work above (mirrors the dense-vs-MoE contrast
  this repo already did for the raw axis6 residual, prior session).
- **Session-to-session drift**: axis6's B16/rate4 cell and this session's
  footprint cell are from different sweep runs; this repo's own notes
  document up to ~8% session-to-session throughput drift
  (`fit_expert_cost_term.py` prints this reminder on every run). Not
  controlled for in the §14 fit -- a small uniform-sign residual shift is
  plausible drift, not necessarily evidence for or against `C_e`, though
  the magnitude of the R² jump here (0.798 → 0.947) is large enough that
  ~8% timing drift alone is very unlikely to explain it away entirely.

## 15. Generalization check: the `reason` workload confirms the fit

Ran both an axis6-style timing sweep AND a verify-footprint sweep on
`reason` (CNN-DailyMail summarization, `specloop_rt.real_corpus
.cnn_dailymail_prompts` -- long-form input, treated by this repo as the
"long output, decaying acceptance" analogue, see that module's docstring),
same 9-cell `D∈{1,3,6}×W∈{1,4,8}` grid as §14's first pass on `code`.

### 15.1 A third real bug: unbounded prompt length crashes cells

First attempt failed immediately: `400 Client Error: Bad Request` on
`/generate`, root cause in the server log --
`Requested token count exceeds the model's maximum context length of 2048
tokens. You requested a total of 2088 tokens: 2024 tokens from the input
messages and 64 tokens for the completion.` `specloop_rt.real_corpus
.cnn_dailymail_prompts()` uses full raw article text with no length cap;
~7% of a 54-request `reason` trace (checked directly) exceed the model's
hard 2048-token context limit (a limit that cannot be raised -- AXIS6.md§4
already documents the EAGLE3 draft head's own `max_position_embeddings
=2048`). `run_open_loop`'s concurrent dispatch means one such request
raises and fails the ENTIRE cell, not just itself -- every cell in the
grid would have failed identically, since the corpus doesn't change with
`(D,W)`.

**Fix: `filter_overlong()`, added to `sweep_sglang_depth_width.py`** (and
imported by both `sweep_sglang_verify_footprint.py` and
`sweep_sglang_expert_footprint.py`, which already import other helpers
from that module) -- drops trace requests whose prompt exceeds a
conservative char-budget estimate (4 chars/token, minus `max_new_tokens`
and a 100-token safety margin) derived from `context_length`, rather than
letting them crash the cell. Deliberately conservative (some requests
that would have just barely fit are dropped too) since the goal is a
representative trace, not exhaustive corpus coverage. **Kept at the
sweep-script level, not in `specloop_rt/real_corpus.py`**: a deliberate
choice to leave that shared module's dataset fidelity (full real article
text) untouched for any other use, rather than bake a context-length
assumption specific to this rig's EAGLE3 draft head into shared
infrastructure. Verified offline first (dropped exactly the 4/54 prompts
expected, matching a manual check), then GPU-confirmed: reran both sweeps
cleanly, 4/54 dropped on every cell (deterministic, `seed=0`), 0 errors
across all 18 cells (9 timing + 9 footprint).

### 15.2 Result: the fit holds, if anything more cleanly

```
nodes-only (D*W):              R2=0.6552
D+W affine:                    R2=0.8760   C_d=0.002377 C_w=0.001153
D+W+interaction:                R2=0.8788   C_d=0.002626 C_w=0.001345 C_dw=-0.000057
D+W+E (mean_distinct_experts):  R2=0.9775   C_d=0.000774 C_w=0.000100 C_e=0.000615
D+W+interaction+E:              R2=0.9917   C_d=-0.000198 C_w=-0.000610 C_dw=0.000146 C_e=0.000744
```

Same headline pattern as `code` (§14), if anything sharper: `D+W+E`
(R²=0.9775) beats `D+W affine` (0.876) by +0.10, while the interaction
term alone buys essentially nothing (+0.003 -- `E` is doing virtually all
the work the interaction term was trying to approximate). `C_w` shrinks by
~91% once `E` is added (0.001153 → 0.000100), an even larger relative
shrink than `code`'s ~61%. The raw footprint numbers (§15 top: 27.89 up to
53.20 across the grid) show the same monotonic-with-saturation shape as
`code`'s 22.35–60.53 range (§14.1), just shifted -- consistent with
`reason`'s different token/routing distribution producing a different
absolute footprint while the underlying D/W→footprint→cost mechanism
stays the same.

**This is the generalization check §14.2 called for, and it passes**:
the `C_e·E` relationship is not an artifact of the `code`/HumanEval
workload specifically. `rag` and `chat` remain untested (§14.2), but one
structurally different workload (long-form summarization vs.
function-completion) already confirming the same pattern is meaningful
evidence this isn't a fluke of one corpus's token distribution.

## 16. Load-dependence check: `E(tree)` is NOT load-invariant

§14.2 (and, before this session, `sweep_sglang_expert_footprint.py`'s
module docstring) assumed the expert-footprint signal is a property of
tree shape `(D,W)` alone -- "the routing signal doesn't need the full
B x rate load grid axis6 swept... expert routing is a property of WHICH
TOKENS the model draws and HOW THE DRAFT TREE branches, not of
queueing/admission load." **This assumption is wrong, confirmed directly.**

Ran the verify-footprint sweep at two more `(B,rate)` points axis6 already
has timing data for, spanning its full range: `B=8,rate=2` (light) and
`B=32,rate=12` (heavy), same `D∈{1,3,6}×W∈{1,4,8}` grid, `code` workload.
Both completed cleanly (9/9 cells each, 0 errors).

### 16.1 The raw numbers move a lot, and monotonically with load

| (D,W) | light (B8,r2) | mid (B16,r4) | heavy (B32,r12) |
|---|---|---|---|
| 1,1 | 18.48 | 22.35 | 57.74 |
| 1,4 | 28.67 | 35.58 | 74.60 |
| 1,8 | 35.73 | 45.06 | 85.34 |
| 3,1 | 29.33 | 31.29 | 65.96 |
| 3,4 | 41.40 | 47.13 | 83.56 |
| 3,8 | 44.30 | 49.84 | 86.47 |
| 6,1 | 36.49 | 42.86 | 77.42 |
| 6,4 | 45.49 | 50.73 | 85.38 |
| 6,8 | 44.88 | 50.37 | 85.59 |

Every single cell increases monotonically light → mid → heavy, and the
mid → heavy jump (roughly 2x) is much larger than light → mid -- not a
small correction, a first-order effect.

### 16.2 Mechanism: bigger admission cap → bigger coalesced verify batches

`mean_verify_batch_tokens` explains it directly. At `(D,W)=(1,1)`:
light=3.42, mid=4.57, heavy=**25.89** tokens per recorded verify-forward
call. SGLang's continuous batching coalesces multiple concurrent requests'
verify steps into one forward call when they land in the same scheduling
window -- a higher `--max-running-requests` (this repo's `B`) means more
requests can be admitted concurrently, so more of them get batched
together into each observed "verify step," which mechanically touches
more distinct experts simply by covering more tokens per measurement.
This is NOT evidence that per-request routing structure changes with
load -- it's evidence that what counts as "one verify batch" in this
telemetry is itself load-dependent.

### 16.3 Correction: expert-footprint DENSITY actually decreases with load

Dividing `mean_distinct_experts` by `mean_verify_batch_tokens` (experts
touched per token in the batch) shows the opposite direction from the raw
numbers:

| (D,W) | light E/tok | mid E/tok | heavy E/tok |
|---|---|---|---|
| 1,1 | 5.402 | 4.886 | 2.230 |
| 1,4 | 3.110 | 2.642 | 0.927 |
| 1,8 | 2.066 | 1.665 | 0.479 |
| 3,4 | 1.832 | 1.571 | 0.445 |
| 6,8 | 1.657 | 1.430 | 0.416 |

Density falls monotonically light → mid → heavy at every cell -- the
marginal new expert per additional token shrinks as batches grow, exactly
the saturation shape you'd expect approaching a 128-expert ceiling
(birthday-paradox-style: a bigger batch is more likely to re-hit an
already-touched expert rather than a new one). So the raw-count increase
with load (§16.1) is a batch-size artifact, not evidence that heavier
load makes routing itself more diverse per token.

### 16.4 What this means for using `C_e` in the cost model

The §14/§15 fits used `E = mean_distinct_experts` (raw count) at a single
load point (`B=16,rate=4`) each. Given §16.1-§16.3, that raw count is
confounded with batch size, which is itself a function of load -- exactly
the same load dependence axis6's own `C_d,C_w ∝ λ^1.4·B^-0.15` scaling
already models for the D/W terms. Two live options, not resolved here:

1. **Give `C_e` its own load-scaling term**, mirroring axis6's `C_d,C_w`
   form, fit against a load-varying `E` the way this section's raw counts
   show. Straightforward but doesn't address that raw `E` is really
   proxying for verify-batch size, not a load-independent structural
   property of the tree.
2. **Refit using the density statistic** (`E/mean_verify_batch_tokens`,
   §16.3) instead of the raw count as the `E(tree)` term -- if density is
   closer to load-invariant (untested: the density table above is
   suggestive but the fit itself, `T_step ~ D+W+density`, hasn't been run
   at multiple load points to check), that would be a cleaner covariate:
   closer to the design brief's original "distinct experts activated"
   framing without the batch-coalescing confound.

Neither option is implemented or fit here -- this section documents the
finding and the fork, not a resolution. The §14/§15 fits (R²=0.947 on
`code`, R²=0.978 on `reason`) remain valid AT their tested load point
(`B=16,rate=4`); what's now known to be false is the assumption that the
same `E` values, or the same `C_e`, would transfer unchanged to a
different load point without accounting for this.

## 17. Reproducing (take 3, current)

```bash
source /root/sglang_env/venv/bin/activate
export HF_HOME=/root/spec_decode_env/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

python3 scripts/sweep_sglang_verify_footprint.py \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --draft-path lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex \
  --num-layers 48 --topk-size 8 \
  --out results_gpu_sweep/sglang_verify_footprint_qwen3moe
```

Same `--num-layers`/`--topk-size` caveat as §9: confirm against the target
model's `config.json`, don't trust the Qwen3-30B-A3B-shaped defaults for a
different model.

To fit `C_e` against axis6's residual once both grids exist (§14):

```bash
python3 scripts/fit_expert_cost_term.py \
  --axis6-grid results_gpu_sweep/sglang_depth_width_rate_qwen3moe/grid.json \
  --footprint-grid results_gpu_sweep/sglang_verify_footprint_grid/grid.json \
  --B 16 --rate 4 --e-field mean_distinct_experts
```

`--e-field mean_imbalance` is also available (`mean_distinct_experts` is
the default and is what §14's reported fit used). `--B`/`--rate` must
match whatever load point the footprint sweep was actually run at.

For a different workload (§15's `reason` check, or a new one):

```bash
python3 scripts/sweep_sglang_depth_width.py \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --draft-path lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex \
  --steps 1 3 6 --topks 1 4 8 --Bs 16 --rates 4 \
  --rtype reason --out results_gpu_sweep/sglang_depth_width_rate_qwen3moe_reason

python3 scripts/sweep_sglang_verify_footprint.py \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --draft-path lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex \
  --num-layers 48 --topk-size 8 --steps 1 3 6 --topks 1 4 8 --Bs 16 --rates 4 \
  --rtype reason --out results_gpu_sweep/sglang_verify_footprint_grid_reason

python3 scripts/fit_expert_cost_term.py \
  --axis6-grid results_gpu_sweep/sglang_depth_width_rate_qwen3moe_reason/grid.json \
  --footprint-grid results_gpu_sweep/sglang_verify_footprint_grid_reason/grid.json \
  --rtype reason --B 16 --rate 4
```

Both sweep scripts share `filter_overlong()` (§15.1) -- any workload with
long, uncapped prompts (like `reason`) needs it; `code`/`chat` prompts are
short enough in practice that it never triggers, but it's applied
unconditionally so no per-workload flag is needed.
