# Axis-8: does Axis-7's expert-footprint mechanism generalize to a
# different MoE architecture (GLM-4.7-Flash)?

## 0. Why this axis exists

AXIS7.md's paper draft (`PAPER_DRAFT_AXIS7.md`) is explicitly scoped to one
MoE model: `Qwen/Qwen3-30B-A3B-Instruct-2507` (128 experts, top-8, ~11%
active). Its own §7 limitations list names this directly: "whether the
expert-footprint mechanism's *magnitude* ... transfers to a MoE with a
different expert count, top-k, or active-param ratio is untested." This
axis is that test, on a different lab's model with a meaningfully
different router config: `zai-org/GLM-4.7-Flash` (`glm4_moe_lite`
architecture, 64 routed + 1 shared experts, top-4, 47 layers).

Model selection ruled out several candidates before landing on GLM
(all checked live, not assumed):

- **Qwen-family anything** (including Qwen3-Coder-30B-A3B, which has a
  confirmed-working SGLang draft) was explicitly excluded — the point was
  architectural diversity, not just a different checkpoint of the same
  router config.
- **gpt-oss-20b** (32 experts, top-4, ~17% active — a genuinely different
  shape) has no usable draft: the only published EAGLE3 head
  (`RedHatAI/gpt-oss-20b-speculator.eagle3`) is in HuggingFace's
  "speculators" checkpoint format (`speculators_model_type` at the
  top level of `config.json`, not a flat `model_type`), which SGLang's
  loader rejects — the same failure mode AXIS6.md already hit with a
  different checkpoint. No SpecForge/SGLang-native draft exists for this
  target at the 20B scale (only a 120B one).
- **Llama-4-Scout-17B-16E** has a real, confirmed-working SpecForge draft
  (`lmsys/SGLang-EAGLE3-Llama-4-Scout-17B-16E-Instruct-SpecForge`) and a
  genuinely different router shape (16 experts, top-1) — but is
  non-viable on a single A100-80GB regardless of quantization: bf16 is
  ~218GB; the one INT4 checkpoint small enough for local disk
  (RedHatAI's W4A16, ~128GB) hits a confirmed, unfixed Marlin MoE kernel
  crash on Ampere/SM80 (`cudaErrorIllegalAddress`, vLLM issue #35922);
  FP8 needs Hopper; the only disk-viable format (Unsloth's GGUF) isn't
  realistically servable via SGLang for MoE. Every SGLang example for
  this model uses 4-GPU tensor parallelism, never single-GPU — Meta's
  "fits on one H100 via int4" claim traces to unreleased sample code, not
  a working checkpoint (confirmed via an SGLang maintainer's "not
  resolved" reply to the exact question in GitHub Discussion #6090).
- **DeepSeek-V2-Lite, Mixtral-8x7B, GLM-4.5-Air, Granite/Arctic/DBRX/
  ERNIE/OLMoE**: no genuine SGLang-native or SpecForge-trained EAGLE3
  draft exists for any of these (checked directly against the
  authoritative `lmsys` HF org listing and general HF/GitHub search).
  DeepSeek-V2-Lite's only candidate draft is a third-party, unverified
  EAGLE-v1-style checkpoint (`num_hidden_layers: 1`,
  `architectures: ["LlamaForCausalLM"]`, no SpecForge provenance).

`zai-org/GLM-4.7-Flash` + `thoughtworks/GLM-4.7-Flash-Eagle3` was the
first candidate that satisfied all constraints simultaneously: genuine
MoE with a different router shape than Qwen3, fits an 80GB A100 in bf16
(~59GB target + 0.3-0.5GB draft), and both target and draft config.json
were verified directly (not from README claims) before committing GPU
time:

- Target: `"architectures": ["Glm4MoeLiteForCausalLM"]`,
  `"model_type": "glm4_moe_lite"`, `n_routed_experts=64`,
  `n_shared_experts=1`, `num_experts_per_tok=4`, `num_hidden_layers=47`,
  not gated.
- Draft: `"architectures": ["LlamaForCausalLMEagle3"]`,
  `"model_type": "llama"` — plain top-level field, **not** speculators
  format — `max_position_embeddings=4096`, not gated.
- SGLang 0.5.17's **released PyPI wheel** (not just GitHub main) ships
  `sglang/srt/models/glm4_moe_lite.py` with the target-side EAGLE3 hooks
  Axis-7's mechanism depends on (`set_eagle3_layers_to_capture`,
  `capture_aux_hidden_states`) — confirmed by unzipping the wheel
  directly, not by trusting the SGLang docs' supported-model list.
- `glm4_moe_lite.py` instantiates the shared SGLang `TopK` class for
  routing (`self.topk = TopK(...)`), the same class whose
  `_post_process_topk_ids` calls `capture_routed_experts_if_allowed()` —
  i.e. it goes through the identical routing-capture path Axis-7's hook
  patches, not a model-specific bypass. Confirmed by reading both files'
  source directly before trusting the hook would fire unmodified.

## 1. Infrastructure notes (worth not re-discovering)

Two real environment issues, unrelated to the model or the mechanism,
cost real time this session:

1. **`/workspace` (this rig's RunPod network volume) has a hard ~26GB
   quota**, despite `df -h` reporting 136TB free (that figure is the
   shared pool's total capacity, not this pod's allocation). A ~59GB
   model download silently failed partway through with
   `OSError: [Errno 122] Disk quota exceeded` once existing usage
   (SGLang venv + partial download) crossed the real ceiling. Binary-
   searched the actual limit directly (`dd` writes of increasing size)
   rather than trusting any single error message. **Fix: put model
   weights on root disk (`/`, 88GB genuinely free on this box) instead**,
   keep large installed packages (the SGLang venv) on `/workspace` if
   they already fit — splitting storage across both filesystems rather
   than moving everything, to avoid an unnecessary ~15GB reinstall.
2. **HuggingFace Hub's newer Xet transfer backend hung indefinitely**
   on this network path, with the progress bar frozen at "1/58 files"
   and the target `.incomplete` blob's mtime not advancing for 7+
   minutes despite the process still being alive (confirmed by polling
   file mtime directly, not just checking the process was alive). No
   error was ever raised. **Fix: `HF_HUB_DISABLE_XET=1` and
   `HF_XET_HIGH_PERFORMANCE=0`, forcing plain HTTPS** — slower
   (~18MB/s measured) but reliable; the modern `hf download` CLI
   (`huggingface-cli download` is deprecated) resumed cleanly on retry.
3. The repo's sweep scripts hardcode `HF_HOME=/root/spec_decode_env/
   hf_cache` (see `sweep_sglang_verify_footprint.py`) rather than reading
   it from the environment — worked around with a symlink
   (`/root/spec_decode_env/hf_cache -> /root/hf_cache`) rather than
   editing the script, so the repo's own reproduction commands stay
   copy-pasteable without an extra env-var override step.

## 2. Pilot: does the expert-footprint hook even fire on `glm4_moe_lite`?

Ran the same first-pass sanity grid AXIS7.md itself used before trusting
the hook (`D∈{1,3,6}×W∈{1,4,8}`, 9 cells, `B=16,rate=4,code`) —
`results_gpu_sweep/glm47_verify_footprint_pilot/grid.json` (first 9
cells; this directory was later extended to the full 24, see §3). All
9/9 cells completed, 0 errors, hook log files non-empty
(3.6-4.8MB/cell of real per-verify-step JSONL rows) on the very first
attempt — the architecture compatibility check in §0 held.

| D | W | mean_distinct_experts | mean_verify_batch_tokens | mean_accept_rate |
|---|---|---|---|---|
| 1 | 1 | 34.87 | 23.3 | 0.626 |
| 1 | 4 | 46.94 | 69.1 | 0.209 |
| 1 | 8 | 48.29 | 111.8 | 0.112 |
| 3 | 1 | 36.95 | — | — |
| 3 | 4 | 50.68 | — | — |
| 3 | 8 | 51.88 | — | — |
| 6 | 1 | 47.11 | — | — |
| 6 | 4 | 49.51 | — | — |
| 6 | 8 | 49.35 | — | — |

Same qualitative shape as Qwen3's original grid: monotonic growth with
both depth and width, with a saturation-near-ceiling non-monotonic step
at the top (here D=6, W=4→8: 49.51→49.35, mirroring Qwen3's own D=6,
W=4→8 near-flat step in AXIS7.md §12).

## 3. Extending to the full 24-cell grid

Extended the same output directory to axis6/7's full
`D∈{1,2,3,4,6,8}×W∈{1,2,4,8}` grid (reusing the sweep script's own
skip-if-done check, so only the 15 new cells ran) —
`results_gpu_sweep/glm47_verify_footprint_pilot/grid.json` now holds all
24 cells, 0 errors.

**Important scope note for anyone re-fitting this data**: a matching
24-cell **baseline** timing grid (hook-disabled, needed to fit `C_e`
against a clean residual the way AXIS7.md §14 does) was **not** built for
GLM — only the 9-cell version exists
(`results_gpu_sweep/glm47_depth_width_pilot/grid.json`). Session
direction pivoted to the full 288-cell load sweep (§4) instead of
completing the 24-cell cost-model generalization check first. The fit
below is therefore run on the 9-cell subset only, not the full 24 —
treat it as a first-pass signal, not a final number.

### 3.1 9-cell cost-model fit: partial confirmation, with a real difference from Qwen3

```
nodes-only (D*W):               R2=0.0458
D+W affine:                     R2=0.4814   C_d=-0.000211 C_w=0.006714
D+W+interaction:                R2=0.8189   C_d=0.011648  C_w=0.015836 C_dw=-0.002737
D+W+E (mean_distinct_experts):  R2=0.7411   C_d=-0.004384 C_w=0.001092 C_e=0.004048
D+W+interaction+E:              R2=0.8700   C_d=0.006363  C_w=0.010487 C_dw=-0.002030 C_e=0.002155
```

`E` does real work here — `D+W+E` (0.741) beats `D+W` affine (0.481) by
+0.26, and `C_w` shrinks substantially once `E` enters (0.0067→0.0011,
an 84% shrink) — the same qualitative signature Axis-7 found on Qwen3.

**But unlike Qwen3, `E` alone does NOT beat the interaction term here.**
`D+W+interaction` (R²=0.819) beats `D+W+E` (R²=0.741) — the opposite of
the Qwen3 paper's headline claim, where `E` beat the interaction term
outright (0.947 vs 0.868, PAPER_DRAFT_AXIS7.md §4). The best fit on this
9-cell subset combines both (`D+W+interaction+E`, R²=0.870).

This could be a real architectural difference (GLM's 64+1/top-4 routing
genuinely interacts with D and W differently than Qwen3's 128/top-8), or
it could be an artifact of the small sample: only 9 cells, 1 seed, and
the raw timing at D=6 is itself non-monotonic in a way D=1 and D=3 are
not (`per_token_s_p50` at D=6: W=1→4→8 is 0.0523→0.0405→0.0404 —
*decreasing* with width, unlike D=1/D=3's clean increases) — the same
kind of small-grid noise AXIS7.md's own §14 flagged before trusting its
first 9-cell pass and extending to 24. **This 9-vs-24 gap is this axis's
biggest open item** — see §6.

## 4. Full 288-cell baseline sweep: does GLM show Qwen3's "rate decides
## it" pattern, or something else?

Given AXIS6.md's own headline finding was established on a 288-cell grid
(not 24), and the user explicitly asked for the matching full grid
rather than a smaller subset, ran
`steps{1,2,3,4,6,8}×topk{1,2,4,8}×B{8,16,24,32}×rate{2,4,12}` — 288
cells, `code`/HumanEval, 30s/cell, 1 seed, hook-disabled (pure timing,
matching AXIS6.md's own baseline convention) —
`results_gpu_sweep/glm47_depth_width_rate_full/grid.json`.

**283/288 cells completed (98.3%), 5 failures**, all isolated to
`steps4_topk1` at various `(B,rate)` combinations
(`B8_rate12`, `B16_rate4`, `B16_rate12`, `B24_rate2`, `B32_rate4`) —
intermittent server-boot stalls (either a 900s startup timeout or a
clean SIGTERM→30s→SIGKILL escalation via the sweep's own
`stop_server()`), not a systematic breakage of that config: other
`steps4_topk1` cells at different loads (e.g. `B8_rate2`, `B16_rate2`,
`B24_rate4`) completed normally in between failures. No resource leak
found (checked open-fd count on the sweep's own process, zombie-process
count, and GPU memory after killing one stuck orphaned server — all
clean); most likely cause is triton MoE-kernel-config autotuning
contention on cold start, since GLM's server logs show "Using default
MoE kernel config... Config file not found" (no cached kernel config for
this model/shape on this session's SGLang install) on every launch.
Genuinely worth fixing before a from-scratch GLM sweep at larger scale,
but did not block or corrupt this run — failed cells are recorded with
their error string and skipped, not silently dropped.

### 4.1 Headline: GLM does NOT replicate Qwen3's "topk=1 dominant regardless of rate" finding

Overall win-share (lowest `per_token_s_p50` wins, across all valid
`(D,B,rate)` groups with ≥2 topk values measured, 72 groups):

| topk | win share |
|---|---|
| 1 | 36.1% |
| 2 | 29.2% |
| 4 | 20.8% |
| 8 | 13.9% |

By rate:

| rate | topk=1 | topk=2 | topk=4 | topk=8 |
|---|---|---|---|---|
| 2 (light) | 33% | 42% | 17% | 8% |
| 4 (medium) | 50% | 21% | 17% | 12% |
| 12 (heavy) | 25% | 25% | 29% | 21% |

By batch cap B:

| B | topk=1 | topk=2 | topk=4 | topk=8 |
|---|---|---|---|---|
| 8 | 33% | 50% | 0% | 17% |
| 16 | 50% | 17% | 17% | 17% |
| 24 | 28% | 28% | 28% | 17% |
| 32 | 33% | 22% | 39% | 6% |

Contrast against AXIS6.md's own tables:

| Model | Architecture | topk=1 win-share by rate |
|---|---|---|
| Dense Llama-3.1-8B | dense | 0% (light) → 46% → 83% (heavy) — clean, monotonic, rate-driven |
| Qwen3-30B-A3B | MoE, 128 experts/top-8 | 83% → 67% → 67% — dominant at every rate tested |
| **GLM-4.7-Flash** | **MoE, 64+1 experts/top-4** | **33% → 50% → 25% — never dominant, non-monotonic** |

GLM is neither a clean rate-driven dense-style pattern nor a
load-invariant MoE-favors-narrow pattern. At heavy load specifically —
exactly where Qwen3 (67%) and dense Llama (83%) both concentrate hardest
on `topk=1` — GLM instead spreads *most* evenly across all four topk
values (25/25/29/21%), the opposite direction from both prior models.

### 4.2 This is a real signal, not noise — spot-checked against raw numbers

At `B=32, steps=4` across rates (an illustrative slice, not
cherry-picked for effect — chosen because it's a mid-depth, high-batch
cell where the sweep has all four topk values at all three rates):

| topk | rate | per_token_s_p50 | mean_accept_rate | mean_accept_length |
|---|---|---|---|---|
| 1 | 12.0 | 0.18757 | 0.299 | 2.197 |
| 2 | 12.0 | **0.16973** | 0.190 | 2.520 |
| 4 | 12.0 | 0.17119 | 0.120 | 2.792 |
| 8 | 12.0 | 0.17377 | 0.120 | 2.796 |

At this cell, `topk=2` genuinely beats `topk=1` at heavy load (0.170s
vs 0.188s per-token, an 11% real difference, not a rounding artifact)
— and the acceptance-rate/accept-length trade moves exactly the way the
underlying mechanism predicts (acceptance falls, accept-length rises
with width). This is a coherent, mechanistically sensible result on its
own terms; it simply lands at a different optimum than Qwen3 did under
the same axes.

## 5. What this means for the Axis-7 paper's generalization claim

`PAPER_DRAFT_AXIS7.md` §7 lists "single MoE model" as an open
limitation and explicitly flags that the mechanism's *magnitude* (not
its existence) transferring to a different expert count/top-k/active-
ratio is untested. This axis's result is genuinely two-sided, and both
halves should go in any paper revision:

- **The expert-footprint mechanism itself partially replicates**: on the
  9-cell subset tested, adding `E` to the cost model produces a real R²
  gain (+0.26) and the same `C_w`-shrinks-once-`E`-enters signature
  Axis-7 uses as its core mechanistic evidence (§3.1). The underlying
  physical story — verify-batch expert footprint costs something real —
  is not Qwen3-specific.
- **But the paper's downstream policy claim does NOT replicate.** Axis-7's
  tree-shape picker (§6 of the paper) is built on Qwen3's specific
  finding that MoE strongly favors narrow trees, and even uses that as
  "reassuring cross-validation" that its fitted cost model rediscovers a
  known result. On GLM, that known result doesn't hold — a picker tuned
  the same way would need to actually run the marginal-utility
  optimization per architecture, not assume convergence to `topk=1`.
  This directly sharpens (not contradicts) the paper's own §7 limitations
  language: the mechanism's *existence* looks architecture-general so
  far; its *quantitative consequences for policy* are not.

## 6. Open items

- **The 9-vs-24-cell fit gap (§3.1) is unresolved** — extend
  `glm47_depth_width_pilot` to the full 24-cell baseline (same grid the
  footprint sweep already covers) and rerun `fit_expert_cost_term.py` on
  the complete match before treating the interaction-beats-E finding as
  settled rather than small-sample noise.
- **The 5 intermittent `steps4_topk1` boot failures (§4)** were not
  root-caused beyond "likely triton kernel-config autotune contention on
  cold start" — worth confirming directly (e.g. pre-warming/caching a
  kernel config per AXIS7.md-style default-config warning) before a
  larger from-scratch GLM sweep, though it did not affect this run's
  validity.
- **No expert-footprint hook run across the 288-cell grid** — §4's result
  is pure timing/acceptance, matching AXIS6.md's own baseline scope
  exactly, but does not by itself explain *why* GLM's optimum differs
  from Qwen3's via the `E(tree)` mechanism the way §3 does at one load
  point. Running the hook across even a subset of the 288-cell grid
  (e.g. a few `(B,rate)` points at each topk) would connect §4's policy
  divergence back to §3's mechanism, the way AXIS7.md §16 connects its
  own load-dependence finding to the mechanism.
- **Single workload (`code`/HumanEval), single seed, one MoE model**
  beyond Qwen3 — same generalization caveats AXIS7.md §7 already states
  apply here too, not resolved by this axis.

## 7. Reproducing

```bash
source /workspace/sglang_env/venv/bin/activate
export HF_HOME=/root/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
# sweep scripts hardcode HF_HOME=/root/spec_decode_env/hf_cache -- symlink
# it to wherever your real cache lives rather than editing the scripts:
mkdir -p /root/spec_decode_env
ln -s /root/hf_cache /root/spec_decode_env/hf_cache

# §2-3: expert-footprint grid (24 cells, B=16 rate=4)
python3 scripts/sweep_sglang_verify_footprint.py \
  --model-path zai-org/GLM-4.7-Flash \
  --draft-path thoughtworks/GLM-4.7-Flash-Eagle3 \
  --num-layers 47 --topk-size 4 \
  --steps 1 2 3 4 6 8 --topks 1 2 4 8 --Bs 16 --rates 4 \
  --context-length 4096 --rtype code \
  --out results_gpu_sweep/glm47_verify_footprint_pilot

# §3.1: matching baseline (currently only 9 of 24 cells built -- see §6)
python3 scripts/sweep_sglang_depth_width.py \
  --model-path zai-org/GLM-4.7-Flash \
  --draft-path thoughtworks/GLM-4.7-Flash-Eagle3 \
  --steps 1 3 6 --topks 1 4 8 --Bs 16 --rates 4 \
  --context-length 4096 --rtype code \
  --out results_gpu_sweep/glm47_depth_width_pilot

python3 scripts/fit_expert_cost_term.py \
  --axis6-grid results_gpu_sweep/glm47_depth_width_pilot/grid.json \
  --footprint-grid results_gpu_sweep/glm47_verify_footprint_pilot/grid.json \
  --B 16 --rate 4

# §4: full 288-cell baseline (steps x topk x B x rate), ~7-8h on one A100
python3 scripts/sweep_sglang_depth_width.py \
  --model-path zai-org/GLM-4.7-Flash \
  --draft-path thoughtworks/GLM-4.7-Flash-Eagle3 \
  --steps 1 2 3 4 6 8 --topks 1 2 4 8 --Bs 8 16 24 32 --rates 2 4 12 \
  --duration 30 --rtype code --context-length 4096 \
  --out results_gpu_sweep/glm47_depth_width_rate_full
```

`--num-layers 47`/`--topk-size 4` are GLM-4.7-Flash's own
`num_hidden_layers`/`num_experts_per_tok` — confirm against the target
model's `config.json` rather than reusing these defaults for a
different model (same caveat AXIS7.md §9/§17 already states).
`--context-length 4096` matches the EAGLE3 draft head's own
`max_position_embeddings` cap, the same constraint pattern AXIS6.md §4
documents for the original Qwen3/Llama draft heads (2048 there).
