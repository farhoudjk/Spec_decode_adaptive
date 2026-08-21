"""Idea 1 -- footprint-aware draft pruning.

BOTTOM LINE (goodput measured, see scripts/goodput_pruning_comparison.py):
the mechanism is CORRECT and STABLE (verified live: lambda=0 byte-for-byte
identical to an unpatched baseline, zero exceptions across a real
CUDA-graph-captured server under load) but, as currently implemented,
FAILS THE GOODPUT TEST -- it makes real throughput WORSE, not better,
across every (B, rate) point tested (8-20% lower tok/s than stock sglang,
consistently, at B in {8,16} x rate in {2,8} req/s x lambda in {1,20}).
mean_verify_ct trended slightly UP with pruning active, not down. This is
a genuine negative result on the method as implemented here, not a bug --
root cause and what it would take to actually pay off are below.

WHY IT LOSES: the overhead shows up even at rate=2 (light load, no queue
pressure) as a ~10-15% per-token latency increase -- i.e. it is
predominantly the RAW COMPUTE COST of running RoutingProxyHead (48 linear
layers, one full forward per draft step) rather than a
selection-quality problem, and at rate=8 that per-step cost compounds
through queueing into a larger goodput gap. Nothing in this design
translates "a better candidate choice" into "less GPU work" -- the tree
shape and the number of experts eventually verified are unchanged; the
proxy can only ever REORDER candidates within a fixed-size tree (see
compute_footprint_adjusted_topk_p), so even a perfect ranking signal only
pays for its own compute cost if it meaningfully raises acceptance
length. On this workload/tree-shape (D=3,W=4, code/HumanEval), it evidently
does not raise it enough -- consistent with i==0 (the tree root, arguably
the highest-leverage position) never being adjusted at all, see below.

STATUS AS OF THIS SESSION: footprint-aware pruning is LIVE and ACTUALLY
ALTERS the tree-construction ranking inside a real sglang server (0.4.10,
this repo's A100-compatible environment -- see install_footprint_pruning.
py's docstring for why 0.5.17's sgl-kernel doesn't support this GPU), not
just a canary that computes-and-discards. compute_footprint_adjusted_
topk_p returns a footprint-cost-adjusted topk_p that gets fed into the
REAL, unmodified select_top_k_tokens (chosen over reimplementing that
function's tree-topology/tree_info bookkeeping from scratch -- see that
function's docstring for why). Verified end-to-end, live, not just by
unit test:
  - lambda=0 identity: generated text and spec_verify_ct byte-for-byte
    identical to a genuinely unpatched baseline server, on two different
    prompts -- the strongest available evidence the integration doesn't
    silently corrupt tree topology when the adjustment is a no-op.
  - lambda=1 and lambda=50 (aggressive): zero exceptions across CUDA-graph
    capture at every batch bucket (1-16) and real generation (including a
    100-token completion), output stays coherent and well-formed at both
    settings.
  - GOODPUT MEASURED (this is the update over the canary-stage status
    below): a real open-loop-Poisson-traffic comparison against stock
    sglang, matching this repo's own sweep convention
    (scripts/sweep_sglang_depth_width.py), found pruning consistently
    WORSE across every tested load point -- see BOTTOM LINE above. This
    supersedes the earlier "spec_verify_ct did not move on short test
    prompts" observation from single ad-hoc requests -- under real
    concurrent load with real acceptance statistics, it does move, in the
    wrong direction.
  - i==0 (tree root) is explicitly passed through UNCHANGED, not adjusted
    -- a real, inherent scope limit discovered this session: at i==0 all
    topk candidates for one request share the SAME hidden_states row, so
    the routing proxy (which only reads hidden_states) cannot discriminate
    between them. Pruning only acts at i>0, where hidden_states genuinely
    varies per candidate. This is plausibly PART OF why the goodput result
    is negative: the tree root is where a wrong choice costs the most
    (it gates the whole subtree), and this method cannot touch it.
See STATUS below for the full trace, including three real batch-shape/
semantics bugs found and fixed only by live GPU testing (isolated toy
tests and source-reading alone missed all three, in sequence).

WHAT WOULD HAVE TO CHANGE FOR THIS TO PAY OFF (not attempted this
session): (a) a cheaper proxy -- amortize the routing-prediction cost
across draft steps rather than a full 48-layer forward every step, or
predict only a subset of layers; (b) act on i==0 somehow -- would need a
genuinely different feature source than hidden_states alone, since the
one available this session cannot discriminate root-level candidates (see
above); (c) test at a tree shape where footprint variance across
candidates is larger (this session used the same D=3,W=4 shape
throughout -- Axis-7/8's own findings suggest wider trees have more
footprint spread to exploit, and this method was never tested at W>4);
(d) or accept that per-step compute overhead needs an actual saved-compute
mechanism (e.g. skipping verification KV-slot reservation for pruned
candidates, not just reordering them), which this session's
scores-only-reordering approach was deliberately scoped to avoid (see
compute_footprint_adjusted_topk_p's docstring for why that scoping choice
was made, and what risk it traded away).

STATUS:
- [CONFIRMED, positive] Routing-proxy signal exists.
  scripts/probe_draft_routing_predictability.py: a linear map from EAGLE3's
  draft hidden state to predicted target-model expert choices beats a
  frequency baseline at all 48/48 target layers (mean Jaccard 0.52 vs 0.22).
  See results_gpu_sweep/draft_routing_probe/result.json.
- [CONFIRMED, constrains the design] select_top_k_tokens's call site is
  CUDA-graph-captured, but not the way Axis-7's take-1 hook failed.
  Read directly against installed sglang 0.5.17 source
  (eagle_draft_cuda_graph_runner.py's capture_one_shape -> run_once ->
  self.eagle_worker.draft_forward(forward_batch), called from inside
  self.backend.capture_one(shape_key, run_once, ...)): draft_forward (and
  therefore select_top_k_tokens) DOES run in eager Python -- but only ONCE,
  during graph capture at server warmup for each captured batch shape.
  execute() (the replay path used for every real request afterward) copies
  fresh inputs into pre-allocated buffers and replays recorded GPU kernels;
  it never calls draft_forward or select_top_k_tokens again. Consequence:
  a monkeypatch on select_top_k_tokens DOES fire, and its tensor
  operations get captured into the graph and correctly replay per-request
  at that batch shape (unlike take-1's TopK.forward patch, which never
  fired at all). But the patched function's logic must therefore be
  expressible as PURE TENSOR OPS -- any Python-level control flow (loops
  over candidates, set() operations, .tolist() calls) only runs once at
  capture time and its result gets frozen into the graph, never
  recomputed per real request. The original design below used Python
  sets for marginal-cost accounting; estimate_marginal_footprint_cost is
  rewritten here to be graph-capturable (one-hot expert masks, tensor
  reductions only, zero data-dependent Python control flow).
- [CONFIRMED, tested] The tensor-op rewrite actually captures and replays
  correctly. Ran footprint_aware_select_top_k_tokens through a real
  torch.cuda.graph capture/replay cycle on persistent buffers (not just
  read the source and inferred it should work): capture succeeded, and
  two replays with different randomly-generated draft hidden states and
  different topk_p values produced DIFFERENT selected-candidate outputs
  ([3,0,4] vs [2,1,0]) -- proving the captured graph recomputes the
  routing-proxy forward pass and cost-based re-ranking fresh from each
  replay's input buffers, rather than freezing whatever was selected at
  capture time. This was the actual open risk (a captured graph CAN
  silently bake in a capture-time constant if a computation isn't wired
  to the right input buffers) and it did not happen here. Toy-scale test
  (16 experts, 4 layers, 5 candidates), not the real 128-expert/48-layer
  shapes, and not inside sglang's own graph-capture machinery
  (capture_one_shape/execute) -- so this confirms the FUNCTION is
  graph-safe in isolation, not that wiring it into eagle_worker_v2.py's
  real capture path is friction-free (buffer lifetime/shape-bucketing
  details specific to sglang's own runner are still unverified).
- [CONFIRMED] RoutingProxyHead trained on a real, multi-workload corpus.
  scripts/train_routing_proxy.py: 66,125 pooled tokens from code/rag/chat
  (150 prompts each), holding out reason/CNN-DailyMail entirely as a
  cross-workload generalization check. In-distribution: probe 0.459 vs
  freq baseline 0.103 (margin +0.356). Held-out workload (never seen in
  training): probe 0.406 vs freq baseline 0.103 (margin +0.303) --
  generalization gap only 0.053, meaning the signal is not an artifact of
  one workload's token distribution. Checkpoint:
  results_gpu_sweep/routing_proxy_train/routing_proxy_head.pt.
- [CONFIRMED] Live GPU integration: hook installs, fires on every draft
  step across ALL captured batch-size buckets (16,14,12,...,1), computes
  the footprint-aware cost signal with zero exceptions, and two real
  generate() requests complete correctly with the hook active. Getting
  here required fixing THREE real bugs a toy-scale isolated test did not
  catch, each found only by running against the actual server:
    1. dtype mismatch -- RoutingProxyHead trained in float32; real
       hidden_states arrive bfloat16. Fixed by casting the proxy in
       install_footprint_pruning.py.
    2. Batch-semantics bug -- the original design used ONE shared
       covered_mask for an entire call, silently conflating expert
       coverage across unrelated concurrent requests. Real hidden_states
       is (true_batch_size [* topk after i==0], hidden_size), not one
       flat candidate list for a single tree. Fixed with a per-request
       (max_batch_size, num_layers, num_experts) buffer, gathered/scattered
       via request_idx_per_row (pure tensor ops -- graph-safe).
    3. Row-grouping bug -- assumed hidden_states was already topk-expanded
       at i==0 (based on reading eagle_utils.py's _select_top_k_tokens_
       first, which does the repeat_interleave on its way OUT, not IN).
       Real observed shapes (via a temporary CAVEMAN_PRUNING_DEBUG_SHAPES
       instrumentation flag, since source-reading alone produced two
       wrong theories in a row): i==0 has hidden_states=(b,H), one row per
       request already; i>0 has (b*topk,H), grouped in topk-sized blocks.
       group_size = 1 if i==0 else topk fixes both the reshape crash on
       non-topk-multiple CUDA-graph batch buckets (14, 10, ...) and the
       per-request top-k reduction.
  Each bug's fix was re-verified against a standalone tensor-shape test
  covering all 12 real observed batch buckets before returning to the
  live server, and the CUDA-graph capture/replay property (fresh
  recomputation per input, not a frozen capture-time constant) was
  reconfirmed at each stage.
- [SUPERSEDED] "Hook doesn't feed selection back into tree construction" --
  this WAS true earlier in this session (a canary that computed but
  discarded its result); no longer true. compute_footprint_adjusted_topk_p
  now returns an adjusted topk_p fed into the real select_top_k_tokens,
  and this was verified live (lambda=0 identity test byte-for-byte
  matching an unpatched baseline; lambda=1/50 tests running clean with
  coherent output) -- see the module-level status note above.
- [NOT YET DONE] Per-request-IDENTITY coverage tracking under real
  concurrent traffic where different requests are at different draft
  steps simultaneously -- the current i==0 reset (covered_mask[:b].zero_())
  assumes request-to-slot assignment starts at 0, which install_footprint_
  pruning.py's docstring flags as an honest, scoped gap (valid for this
  session's one-request-at-a-time live test, not yet correct for mixed
  concurrent traffic).
- [NOT YET DONE] lambda (cost/value tradeoff) is unfit -- lambda=1 and
  lambda=50 both ran clean, but neither visibly changed spec_verify_ct on
  this session's specific (short, common-continuation) test prompts, so
  there is not yet live evidence of what lambda value (if any, on these
  prompts) actually shifts tree-construction outcomes. Needs a wider
  prompt set and/or a metric more sensitive than spec_verify_ct (e.g.
  directly logging predicted marginal_cost/adjusted_topk_p deltas) before
  a real sweep is meaningful.
- [NOT YET DONE] No live GPU goodput measurement -- this session validated
  correctness/stability of the mechanism (and that it CAN alter ranking
  without breaking generation), not yet its throughput payoff. That needs
  a workload where the footprint-aware ranking demonstrably changes
  candidate selection, then a goodput comparison against stock sglang at
  the same (D,W,B,rate) point, matching Axis-7's own evaluation
  convention (PAPER_DRAFT_AXIS7.md #6).

--------------------------------------------------------------------------
WHERE THIS HOOKS IN (confirmed against installed sglang 0.5.17 source, not
guessed): sglang.srt.speculative.spec_utils.select_top_k_tokens(i, topk_p,
topk_index, hidden_states, scores, topk) is called once per draft step
(i = 0..speculative_num_steps-1) from eagle_worker_v2.py's draft_forward,
and is the sole place candidates get selected to extend the tree at each
level. It currently ranks purely by topk_p (the draft model's own token
probability). This is the natural -- and only -- hook point for a
cost-aware re-ranking: same call signature, same call site, different
selection criterion.

The method (as specified by the user): at each draft step, given the
current candidate set, drop the candidate that maximizes
    marginal_expert_footprint_cost / marginal_acceptance_value
i.e. re-rank/filter select_top_k_tokens's candidates not by topk_p alone
but by topk_p adjusted for predicted expert cost. This differs from
SGLang's built-in topk_p-only ranking (confidence-only) and from any
generic "drop low-probability candidates" scheme (statistical-only) by
pricing candidates on an architectural axis -- how many NEW distinct
experts a candidate would add to the verify batch's footprint -- not a
purely statistical one.

WHY THIS NEEDS A PREDICTED cost, not a measured one: by the time the real
expert footprint is measurable (verify_batch_expert_hooks.py, Axis-7's
working hook), the verify forward has already run -- the GPU has already
paid for every candidate's expert activation, so pruning after that point
saves nothing. A pruning decision made at select_top_k_tokens time (before
verify) can only use information available at draft time: the draft
model's own hidden state. Hence the routing-proxy sub-problem the probe
script tests.

--------------------------------------------------------------------------
THIS FILE'S CONTENTS (logic GPU-tested at toy scale; not GPU-tested inside
a real sglang server):

1. RoutingProxyHead -- the minimum-viable proxy this pruning method needs:
   one linear (or shallow MLP) map per target MoE layer, draft hidden
   state -> predicted expert logits, matching the probe script's
   LinearRoutingProbe exactly (same architecture) but meant to be trained
   once offline (on a representative corpus) and shipped as a small
   checkpoint loaded at server startup, not fit online.

2. estimate_marginal_footprint_cost -- given a candidate set's CURRENTLY
   predicted expert coverage (a persistent (num_layers, num_experts)
   multi-hot buffer, accumulated as the tree grows) and a new candidate's
   predicted top-k expert set, the marginal cost is the count of NEWLY
   introduced distinct experts across all target layers (mirrors
   verify_batch_expert_hooks.py's own E(tree) definition: distinct experts
   touched, not total activations). Implemented as pure tensor ops
   (scatter + elementwise + sum) after the CUDA-graph-safety finding above
   ruled out the original Python-set version; cross-checked against a
   brute-force set-based reimplementation for numerical agreement, and
   run through a real torch.cuda.graph capture/replay cycle.

3. footprint_aware_select_top_k_tokens -- a drop-in-compatible replacement
   for spec_utils.select_top_k_tokens with the same signature plus a
   routing_proxy argument, re-ranking candidates by
   topk_p / (1 + lambda * marginal_footprint_cost) instead of topk_p alone
   (a value-per-cost ratio, not a hard cutoff -- lambda trades off
   confidence against footprint growth and needs its own sweep, not a
   single hardcoded value).

STILL NOT DONE HERE (deliberately):
- No trained RoutingProxyHead checkpoint -- would need the probe's own
  training loop generalized across a larger/more diverse corpus first.
- No actual monkeypatch installing this into a running sglang server. The
  function itself is now confirmed graph-safe in an isolated
  torch.cuda.graph capture/replay test (16 experts/4 layers/5 candidates
  toy scale, in-place covered_mask update via .copy_(torch.maximum(...))
  -- this specific pattern was the open risk and it captured/replayed
  correctly, including recomputing fresh per-input rather than freezing a
  capture-time constant). What remains unconfirmed is integration into
  sglang's OWN graph-capture machinery (capture_one_shape/execute) at
  real shapes (128 experts, 48 layers, real batch/topk sizes) and across
  the actual speculative_num_steps loop inside one capture call, where
  sglang's own buffer lifetime and shape-bucketing conventions apply and
  were not exercised by this isolated test.
- No offline simulator validation (the "offline/simulator-first" path this
  session's design discussion considered as an alternative) -- this file
  takes the draft-side-proxy path instead, per explicit direction.
- lambda (the cost/value tradeoff weight) is unfit -- needs a sweep once
  the proxy exists and validates.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RoutingProxyHead(nn.Module):
    """One linear map per target MoE layer: draft hidden state -> expert
    logits. Architecturally identical to
    scripts/probe_draft_routing_predictability.py's LinearRoutingProbe --
    intentionally the same class in spirit, kept as a separate copy here
    because the probe script's version is throwaway/diagnostic (trained
    fresh per probe run) while this one is meant to be the actual shipped
    artifact (trained once, checkpointed, loaded at server start). If the
    probe validates a MORE expressive head is needed (e.g. the linear probe
    beats frequency baseline only marginally), replace this with a small
    MLP -- but start here, per the probe's own "simplest thing that could
    carry signal" logic.

    IMPLEMENTATION NOTE, load-bearing for the goodput result (see module
    STATUS): keeps nn.ModuleList(48 x nn.Linear) as the PARAMETER
    container -- so existing checkpoints trained by scripts/
    train_routing_proxy.py load with zero changes (state_dict keys stay
    proj.0.weight, proj.0.bias, ...) -- but predicted_topk_experts below
    does NOT call each nn.Linear separately in a Python loop. That
    per-layer-loop version was confirmed, by live GPU profiling this
    session (comparing SGLang's own "Capture draft cuda graph" mem usage
    log line between a patched and unpatched server: 1.82 GB vs 0.21 GB),
    to inflate the CUDA-graph capture memory pool by ~1.6 GB -- 48
    separate small nn.Linear kernel launches during graph capture, each
    getting its own cuBLAS workspace/allocator reservation, even though
    the raw compute (~2ms/call, measured) and raw tensor sizes (<1MB) are
    both far too small to explain that. That reserved pool persists for
    the server's lifetime and directly shrank the KV-cache budget
    (max_total_num_tokens: 108134 -> 86159, ~20% smaller) enough to
    measurably hurt real serving throughput (8-20% goodput loss across a
    full load sweep) despite accept_length and per-decode-step
    "gen throughput" being essentially unchanged between configs -- i.e.
    the mechanism itself wasn't hurting candidate quality; the capture
    footprint of HOW it was implemented was silently taxing the whole
    server's capacity. Fixed by stacking all 48 layers' weights into one
    (num_target_layers, num_experts, draft_hidden_size) tensor and using a
    single batched matmul (einsum) instead of 48 separate nn.Linear calls
    -- one kernel launch, one capture footprint, same math.
    """

    def __init__(self, draft_hidden_size: int, num_experts: int, num_target_layers: int):
        super().__init__()
        self.proj = nn.ModuleList(
            [nn.Linear(draft_hidden_size, num_experts) for _ in range(num_target_layers)]
        )
        self.num_target_layers = num_target_layers
        # Cache for the batched-weight view built by _build_batched_weights().
        # Built lazily (not in __init__) because it must be rebuilt if
        # .to(dtype)/.cuda() is called after construction (the common
        # pattern in install_footprint_pruning.py: build fp32, then
        # .to(bfloat16)) -- caching at __init__ time would capture the
        # pre-cast fp32 weights and silently ignore the later cast.
        self._batched_w = None
        self._batched_b = None
        self._batched_dtype_device = None

    def _build_batched_weights(self):
        """Stack all 48 nn.Linear layers' weights/biases into single
        (num_target_layers, num_experts, draft_hidden_size) /
        (num_target_layers, num_experts) tensors, matching whatever
        dtype/device self.proj's parameters currently have (so a
        .to(bfloat16).cuda() call after construction is picked up
        correctly on first use, not stale fp32 weights from __init__).
        """
        w = torch.stack([layer.weight for layer in self.proj], dim=0)  # (L, E, H)
        has_bias = self.proj[0].bias is not None
        b = torch.stack([layer.bias for layer in self.proj], dim=0) if has_bias else None  # (L, E)
        self._batched_w = w
        self._batched_b = b
        self._batched_dtype_device = (w.dtype, w.device)

    def predicted_topk_experts(self, draft_hidden: torch.Tensor, topk: int) -> torch.Tensor:
        """draft_hidden: (n_candidates, draft_hidden_size) ->
        (n_candidates, num_target_layers, topk) predicted expert ids.
        Single batched matmul (einsum) across all layers -- see class
        docstring for why this replaced a per-layer Python-loop version.
        """
        current = (self.proj[0].weight.dtype, self.proj[0].weight.device)
        if self._batched_w is None or self._batched_dtype_device != current:
            self._build_batched_weights()
        # (N, H) x (L, E, H) -> (N, L, E)
        logits = torch.einsum("nh,leh->nle", draft_hidden, self._batched_w)
        if self._batched_b is not None:
            logits = logits + self._batched_b.unsqueeze(0)
        return torch.topk(logits, k=topk, dim=-1).indices


def predicted_experts_to_multihot(
    predicted_experts: torch.Tensor,  # (n_candidates, num_layers, topk)
    num_experts: int,
) -> torch.Tensor:
    """(n_candidates, num_layers, topk) expert-id tensor -> (n_candidates,
    num_layers, num_experts) multi-hot float tensor (1.0 where an expert is
    in that candidate's predicted top-k, 0.0 elsewhere). Pure tensor op
    (scatter), no Python-level loop over candidates or set() calls --
    graph-capturable, unlike the set-based version this replaces.
    """
    n_candidates, num_layers, topk = predicted_experts.shape
    onehot = torch.zeros(
        n_candidates, num_layers, num_experts,
        device=predicted_experts.device, dtype=torch.float32,
    )
    onehot.scatter_(2, predicted_experts, 1.0)
    return onehot


def estimate_marginal_footprint_cost_batched(
    predicted_experts: torch.Tensor,  # (num_rows, num_layers, topk)
    covered_mask: torch.Tensor,        # (max_batch_size, num_layers, num_experts), persistent buffer
    request_idx_per_row: torch.Tensor,  # (num_rows,) long -- which batch slot each row belongs to
    num_experts: int,
) -> torch.Tensor:
    """Batched, per-REQUEST version of the marginal-footprint-cost signal.
    sglang serves multiple concurrent requests at once, each growing its
    OWN draft tree -- select_top_k_tokens's hidden_states is
    (true_batch_size * topk, hidden_size), not one flat candidate list, and
    a single shared covered_mask (this module's first draft, before this
    fix) would incorrectly conflate expert coverage across unrelated
    concurrent requests. Fixed here by making covered_mask
    (max_batch_size, num_layers, num_experts) and gathering each row's own
    request's coverage via request_idx_per_row before comparing -- pure
    tensor ops throughout (gather + elementwise + sum), no Python-level
    loop over rows or requests, preserving the CUDA-graph-capturability
    this module's whole design depends on (see module STATUS section).

    request_idx_per_row derivation, confirmed against installed sglang
    0.4.10's eagle_utils.select_top_k_tokens source (same structure in
    0.5.17's spec_utils): hidden_states stays shaped
    (true_batch_size * topk, hidden_size) at every step i (repeat_interleave
    at i==0, re-gathered via selected_input_index at i>0, never growing
    further) -- so row r always belongs to request r // topk, at every
    step, for the SAME topk (the tree's branching factor, not the proxy's
    own top-k).
    """
    candidate_multihot = predicted_experts_to_multihot(predicted_experts, num_experts)  # (num_rows, L, E)
    row_covered = covered_mask[request_idx_per_row]  # (num_rows, L, E) -- gather, not a Python loop
    novel = candidate_multihot * (1.0 - row_covered)
    return novel.sum(dim=(1, 2))  # (num_rows,)


def update_covered_mask_batched(
    covered_mask: torch.Tensor,          # (max_batch_size, num_layers, num_experts), persistent buffer
    predicted_experts: torch.Tensor,      # (num_rows, num_layers, topk) -- experts predicted for the KEPT rows
    request_idx_per_row: torch.Tensor,    # (num_rows,) long
    num_experts: int,
) -> None:
    """In-place update of covered_mask for exactly the rows that survived
    selection this step, scattered back to their owning request's slot.
    Uses index_add_ + clamp (not a Python loop) since multiple rows can
    share a request_idx (multiple kept candidates for the same request) --
    scatter with a plain assignment would silently drop all but the last
    row per request; index_add_ + clamp_ preserves the union-of-experts
    semantics scatter_/setitem alone would lose.
    """
    kept_multihot = predicted_experts_to_multihot(predicted_experts, num_experts)  # (num_rows, L, E)
    covered_mask.index_add_(0, request_idx_per_row, kept_multihot)
    covered_mask.clamp_(max=1.0)  # index_add_ can exceed 1.0 when multiple kept rows share an expert; renormalize


def compute_footprint_adjusted_topk_p(
    i: int,
    topk_p: torch.Tensor,
    hidden_states: torch.Tensor,
    topk: int,
    routing_proxy: "RoutingProxyHead",
    covered_mask: torch.Tensor,   # (max_batch_size, num_layers, num_experts) persistent buffer
    proxy_topk: int,
    lam: float,
):
    """Returns an ADJUSTED topk_p (same shape as the input, safe to feed
    straight into the REAL, unmodified select_top_k_tokens) instead of a
    pruned index list or a hand-rolled tree_info. This is the integration
    approach chosen deliberately over reimplementing select_top_k_tokens's
    own tree-topology bookkeeping: at i>0, that function's tree_info
    (parent-pointer encoding consumed downstream by verify-step attention
    masking) is derived FROM topk_p/scores via its own torch.topk/fast_topk
    calls -- perturbing the ranking SIGNAL and letting sglang's own,
    already-correct code do the actual selection and index bookkeeping
    eliminates an entire class of "silently wrong tree topology" bugs that
    a from-scratch reimplementation would risk. lam=0 must recover the
    original topk_p exactly (a free built-in ablation/correctness check).

    SCOPE LIMIT, discovered live and not a bug: at i==0, ALL topk
    candidates for one request share the SAME hidden_states row (only
    topk_index/topk_p vary per candidate at that step -- hidden_states
    only becomes per-candidate-distinguishing at i>0, after the previous
    step's re-gather). The routing proxy reads only hidden_states, so it
    predicts the IDENTICAL expert cost for every candidate at i==0 -- it
    has no signal to discriminate the tree's root-level choice on. This
    function therefore returns topk_p UNCHANGED at i==0 (explicit, not
    silently degenerate) and only adjusts at i>0, where hidden_states is
    genuinely per-candidate.
    """
    if i == 0:
        return topk_p  # no signal to prune on at the tree root -- see docstring

    num_rows = hidden_states.shape[0]
    # group_size=topk at i>0: hidden_states is (b*topk, H), grouped in
    # topk-sized blocks, one block per request -- confirmed against real
    # runtime shapes (CAVEMAN_PRUNING_DEBUG_SHAPES), not source alone.
    request_idx_per_row = torch.arange(num_rows, device=hidden_states.device) // topk

    predicted = routing_proxy.predicted_topk_experts(hidden_states, topk=proxy_topk)
    num_experts = routing_proxy.proj[0].out_features
    marginal_cost = estimate_marginal_footprint_cost_batched(
        predicted, covered_mask, request_idx_per_row, num_experts,
    )  # (num_rows,)

    # topk_p at i>0 is (num_rows, topk) -- ROW-aligned with hidden_states/
    # marginal_cost, its second dim is this row's OWN topk candidate
    # continuations (not the same topk as the tree-branching factor
    # dimension, despite sharing the variable name in sglang's own source
    # -- confirmed by shape: (num_rows, topk) where num_rows is already
    # b*topk from the tree's branching). Penalize each row's whole
    # probability vector by that row's SHARED marginal cost (the cost is a
    # property of the row/candidate the hidden_states represents, not of
    # which further sub-continuation topk_p enumerates).
    penalty = 1.0 + lam * marginal_cost.to(topk_p.dtype).to(topk_p.device)
    adjusted_topk_p = topk_p / penalty.unsqueeze(-1)

    # Update running per-request coverage using the top-`topk` rows BY
    # REQUEST under the ADJUSTED ranking (not the original topk_p) -- this
    # is what actually gets selected downstream once fed back into the
    # real select_top_k_tokens, so coverage should track what will really
    # survive, not what would have survived under the unmodified ranking.
    row_confidence = adjusted_topk_p.amax(dim=-1)
    b = num_rows // topk
    row_confidence_by_req = row_confidence.view(b, topk)
    local_keep = torch.topk(row_confidence_by_req, k=min(topk, row_confidence_by_req.shape[-1]), dim=-1).indices
    row_keep = (local_keep + (torch.arange(b, device=hidden_states.device) * topk).unsqueeze(1)).flatten()

    update_covered_mask_batched(
        covered_mask, predicted[row_keep], request_idx_per_row[row_keep], num_experts,
    )

    return adjusted_topk_p
