# Axis-6: SGLang EAGLE3 tree width, and why the pilot's finding didn't survive scale

## 0. Why SGLang, not a vLLM patch

Axis-5 found that vLLM 0.9.2 has no tree-style speculative decoding at all —
`EagleProposer` only ever supports a linear chain via `num_speculative_tokens`
(depth, γ), with no branching-factor config field, no tree-attention mask
construction. Checked whether this landed in any later vLLM release (up to
0.26.0, current at session time): no. GitHub issue #18327 requesting it was
closed "not planned."

SGLang has real tree-width support already built and documented:
`--speculative-num-steps` (depth) and `--speculative-eagle-topk` (width,
branching factor per step) are independent, both configurable, both
launch-time-static — width is not live-adjustable in SGLang either; confirmed
from reading `adaptive_spec_params.py`'s `adaptive_unsupported_reason()`,
which hard-refuses `--speculative-adaptive` unless `eagle_topk in (None, 1)`.
Building tree-attention into vLLM ourselves would have been a deep
architectural change (new mask construction, new KV layout, new verify-step
logic) for a mechanism SGLang already has, so this axis moved to SGLang
(v0.5.16) rather than patching vLLM further.

## 1. Setup

Two model pairings, both validated live end-to-end (`spec_accept_rate`,
`spec_accept_length` visible per-request in SGLang's native response
telemetry — no custom patch layer needed, unlike vLLM's freeze-bug
workaround from Axis-5):

| Role | Target | Draft (EAGLE3) |
|---|---|---|
| Dense | `meta-llama/Llama-3.1-8B-Instruct` (100% active params) | `jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B` |
| MoE | `Qwen/Qwen3-30B-A3B-Instruct-2507` (3.3B/30.5B active, ~11%) | `lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex` |

Both draft checkpoints had to be the SGLang-repackaged versions, not the
original author's (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` and
`RedHatAI/gpt-oss-20b-speculator.eagle3` both failed — see §4 for the exact
errors and why). Mixtral-8x7B (Axis-5's MoE target) has no published EAGLE
draft checkpoint anywhere, any framework — ruled out early, not a
config/engine problem. FP8 quantization is also unavailable on this hardware
for MoE kernels specifically: this box is an A100 (SM80/Ampere), and
SGLang's Triton fused-MoE kernel requires `fp8e4nv`, which needs Hopper
(SM90+) — confirmed via `nvidia-smi --query-gpu=compute_cap` and the
resulting `CompilationError`. Both models run bf16.

## 2. Two sweep rounds — the first one was wrong

### 2.1 First round: 48 cells, no arrival-rate axis, WRONG conclusion

Initial sweep (`scripts/sweep_sglang_depth_width.py`, first version) fired a
fixed-size synchronous batch of `B` requests per cell (B∈{8,32}), varying
depth (steps∈{1,3,5}) and width (topk∈{1,2,4,8}). Result, published as the
headline finding: **"width monotonically hurts dense at every depth; width
only helps MoE once depth is deep enough to amortize verify cost."**

This did not hold at scale (§3). The synchronous-batch design had no
explicit load/arrival-rate axis — it measured one specific, unlabeled load
regime and the result was mistakenly generalized as if it depended on model
architecture.

### 2.2 Second round: 576 cells, real open-loop Poisson arrivals

Rebuilt the harness with an actual arrival-rate axis, reusing
`specloop_rt.workload.homogeneous` — the same Poisson-arrival trace
generator this repo's vLLM harness (`specloop_rt.replay`) already uses for
its own `--rate` flag, so "rate" here means the same thing it means there: a
client-side arrival intensity (req/s), independent of the batch admission
cap. `B` was redefined to mean `--max-running-requests` (the server's
admission ceiling), not a synchronous batch size — matching this repo's own
`admit_kw.max_num_seqs` convention. Requests dispatch paced to their
Poisson-drawn `arrival_s` timestamp via a thread pool sized to never be the
client-side bottleneck, so the server's own admission cap is the only real
throttle shaping the result.

Grid: `steps{1,2,3,4,6,8} × topk{1,2,4,8} × B{8,16,24,32} × rate{2,4,12
req/s}` — 288 cells × 2 models = 576 total, 30s trace duration per cell,
`code`/HumanEval workload, 1 seed. **576/576 cells completed, 0 errors.**

Two real bugs found and fixed while building this (both in
`scripts/sweep_sglang_depth_width.py`, comments in the code explain the
mechanism):

1. `num_draft_tokens` must satisfy `num_draft_tokens - 1 <= steps * topk`
   (the tree-organize step's `torch.topk` call crashes with `RuntimeError:
   selected index k out of range` otherwise) — clamped per cell:
   `min(requested, steps*topk + 1)`.
2. Dtype auto-detection mismatched on the Llama+EAGLE3 pairing specifically
   (a fused RMSNorm CUDA kernel raised `ValueError: Mismatched Tensor ...
   expected dtype=bfloat16`) — fixed by passing `--dtype bfloat16` explicitly
   rather than relying on SGLang's auto-detection from the target config.

## 3. Result: rate decides it, not the model

Collapsing every (depth, batch, rate) combination to "which topk had the
lowest per-token latency" (72 groups per model):

**Overall win share, all 576 cells:**

| topk | MoE | Dense |
|---|---|---|
| 1 | **72%** | 31% |
| 2 | 3% | 21% |
| 4 | 7% | 26% |
| 8 | 18% | 22% |

MoE stays `topk=1`-dominant across the board. Dense looks close to a
four-way split until broken out by rate:

**Dense win share by rate:**

| rate | topk=1 | topk=2 | topk=4 | topk=8 |
|---|---|---|---|---|
| 2 (light) | **0%** | 0% | 54% | 46% |
| 4 (medium) | 8% | 58% | 21% | 12% |
| 12 (heavy) | **83%** | 4% | 4% | 8% |

**Dense's optimal width is a clean function of load.** At light load
(rate=2), `topk=1` never wins a single cell — width 4 or 8 is faster at
every depth tested, because the GPU has idle compute the extra verify work
slots into for free (the same "memory-bound, speculation nearly free"
regime the Axis-5 roofline theory describes for depth). At heavy load
(rate=12), `topk=1` wins 83% of cells — the same extra verify work now
competes with a real queue and every branch costs real time. Concretely, at
B=32 steps=4: `topk=1`'s per-token latency barely moves between rate=2 and
rate=4 (6.96ms → 6.98ms) while `topk=8`'s nearly doubles over the same
change once rate reaches 12 (7.0ms → ~28ms extrapolated from the B=32
rate=12 table in `results_gpu_sweep/sglang_depth_width_rate_llama/grid.json`).

**MoE win share by rate:**

| rate | topk=1 | topk=2 | topk=4 | topk=8 |
|---|---|---|---|---|
| 2 | 83% | 0% | 8% | 8% |
| 4 | 67% | 0% | 8% | 25% |
| 12 | 67% | 8% | 4% | 21% |

MoE never drops below 67% `topk=1` win-share at any rate — the earlier
pilot's "fastest cell in the whole grid is steps=5,topk=8" does not
replicate. At the same depth and rate scaled to the full grid, `topk=1` is
faster or tied at every batch size tested. Batch cap (B) has a much weaker
effect than rate on both models: MoE's `topk=1` win-share moves only 6
points across B=8→32 (67%→78%), dense moves more (17%→39%) but still far
less than rate's 0%→83% swing.

**Mechanism, not just correlation.** Acceptance rate falls hard as width
grows on both models (dense steps=3: 40% at topk=1 → 12% at topk=8) — the
same acceptance-collapse curve Axis-5 found for depth alone. But
`mean_accept_length` keeps climbing with width even as rate falls (dense
steps=3: 2.19 → 2.76 tokens/step), because a wider tree offers more
candidate paths per verify pass. Whether that trade wins depends entirely on
whether the GPU has slack to spend on it — the same load-boundedness
mechanism the roofline theory already gives for depth, now shown to apply
identically to width.

## 4. Checkpoint compatibility notes (worth not re-deriving)

- `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (the original author's Llama EAGLE3
  checkpoint, works fine on vLLM) fails on SGLang with `Parameter d2t not
  found in params_dict` / `AssertionError: self.org_vocab_size=128256 ...
  loaded_weight.shape[output_dim]=32000` — different parameter naming/vocab
  convention than SGLang's loader expects. Use
  `jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B` instead (SGLang's own docs
  cite this exact repo for this exact target).
- `RedHatAI/gpt-oss-20b-speculator.eagle3` (Red Hat's "speculators"-format
  packaging) fails on SGLang with `ValueError: Unrecognized model ... Should
  have a model_type key in its config.json` — the config is nested under
  `speculators_config`/`transformer_layer_config` rather than a flat
  top-level `model_type`, which SGLang's config loader doesn't parse. This
  is a systemic incompatibility with the whole `speculators`-format
  checkpoint family (confirmed via `sgl-project/sglang#8229` and `#18216`
  hitting the identical error on different checkpoints), not specific to
  gpt-oss. No fix attempted — abandoned gpt-oss-20b as a candidate MoE
  target in favor of Qwen3-30B-A3B, which has a SpecForge-trained,
  SGLang-native checkpoint already.
- Both EAGLE3 draft heads (Llama and Qwen3-MoE) declare
  `max_position_embeddings=2048` — `--context-length` must be ≤2048 or the
  server fails at config-resolution time with `ValueError: Target model's
  context_length (N) is greater than the derived context_length (2048)`.

## 5. Reproducing this session's results

```bash
python3 -m venv /root/sglang_env/venv && source /root/sglang_env/venv/bin/activate
pip install "sglang[all]"
export HF_TOKEN=<token with meta-llama/Llama-3.1-8B-Instruct access>
export HF_HOME=/root/spec_decode_env/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# MoE grid (~4-5h on one A100-80GB)
python3 scripts/sweep_sglang_depth_width.py \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --draft-path lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex \
  --steps 1 2 3 4 6 8 --topks 1 2 4 8 --Bs 8 16 24 32 --rates 2 4 12 \
  --duration 30 --rtype code --max-new-tokens 128 --port 30001 \
  --out results_gpu_sweep/sglang_depth_width_rate_qwen3moe

# Dense grid (~4-5h)
python3 scripts/sweep_sglang_depth_width.py \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
  --steps 1 2 3 4 6 8 --topks 1 2 4 8 --Bs 8 16 24 32 --rates 2 4 12 \
  --duration 30 --rtype code --max-new-tokens 128 --dtype bfloat16 --port 30001 \
  --out results_gpu_sweep/sglang_depth_width_rate_llama
```

`results_gpu_sweep/sglang_depth_width_rate_*/grid.json` (576 cells total,
0 errors) are force-added past this repo's `results_gpu*/` gitignore rule,
same convention as every prior axis. Per-cell `server_<tag>.log` files are
not committed (bulky, regenerable). The interactive report at
`reports/eagle3_depth_width.html` (self-contained, embeds both full grids)
renders both models' depth×width heatmaps filterable by batch cap and
arrival rate.

## 6. Open items

- Earlier pilot data (`sglang_depth_width_qwen3moe/`,
  `sglang_depth_width_llama/`, 48 cells, no rate axis) is kept in the repo
  for provenance but its headline conclusion is superseded by §3 — do not
  cite it.
- Only 1 seed throughout; multi-seed replication unconfirmed.
- Only `code`/HumanEval workload tested; other request types (`rag`, `chat`,
  `reason`) from `specloop_rt.workload`/`real_corpus` are wired into the
  harness (same `--rtype` flag this repo's vLLM sweeps use) but unrun here.
- MoE+EAGLE on a genuinely sparse target with a lower active-parameter ratio
  than Qwen3-30B-A3B's ~11% (e.g. a bigger MoE, if a SpecForge-trained draft
  head existed and fit this hardware) would sharpen whether MoE's persistent
  `topk=1` preference is architecture-driven or an artifact of this specific
  model's routing.
