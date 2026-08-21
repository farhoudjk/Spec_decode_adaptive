"""Idea-1 prerequisite probe: is target-model MoE routing predictable from
the EAGLE3 draft head's own hidden state, before the target verify pass runs?

Why this exists: the footprint-aware-pruning idea needs a per-candidate cost
signal AT DRAFT TIME (before verify), to decide which candidates to drop.
Axis-7's only working expert-footprint signal (verify_batch_expert_hooks.py)
reads routing AFTER the target verify forward already ran -- by
construction, too late to use as a pruning signal for that same verify
pass. This script tests the actual sub-problem a draft-side routing proxy
depends on: does the EAGLE3 draft hidden state carry enough signal about
which experts the TARGET model will route a token to, that a lightweight
probe head could predict it before paying for verification?

Method: for real prompts, run the target model's teacher-forced forward
pass with output_router_logits=True to get ground-truth top-k expert IDs
per layer per token. Separately extract the target's own aux hidden states
(the layers EAGLE3 training normally concatenates) and feed them through
the EAGLE3 draft head to get the draft's hidden state at each position.
Fit one linear probe per target MoE layer (draft hidden state -> per-expert
logit, multi-label) on a train split, evaluate top-k set overlap (Jaccard
against the true top-k expert set) on a held-out split, against two
baselines: (a) uniform random top-k, (b) global expert-frequency top-k
(the "always guess the most popular experts" baseline -- routing is
somewhat frequency-skewed even under a load-balancing aux loss, so this is
the real baseline to beat, not chance).

If the probe cannot beat the frequency baseline by a real margin, footprint-
aware pruning has no signal to prune on and should not be built. This
script's whole job is answering that question cheaply before committing to
the pruning pipeline.

Position shift, load-bearing for correctness: EAGLE's draft hidden state at
position t is built from embeds[t] (current token embedding) and aux[t]
(target aux features at t), and is used to PROPOSE the token at position
t+1 -- so it is compared against the target's true routing at t+1, not at
t. Getting this backwards would let the probe trivially "predict" routing
it was directly handed via aux[t]'s own information content, inflating the
result meaninglessly.

Does not require a running SGLang server -- plain `transformers` for the
target model (teacher-forced, no sampling, no draft-tree construction) plus
a standalone port of SGLang's own `LlamaForCausalLMEagle3` layer math for
the draft head, factored into
specloop_rt/sglang_patch/eagle3_draft_port.py (shared with
scripts/train_routing_proxy.py so both use identical feature extraction).
See that module's docstring for why the draft checkpoint isn't loadable
via plain `AutoModelForCausalLM`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)

from specloop_rt.sglang_patch.eagle3_draft_port import (
    get_target_aux_layer_ids,
    load_eagle3_draft_layer,
)

TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DRAFT_MODEL = "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex"

HF_HOME_DEFAULT = "/root/hf_cache"


@dataclass
class ProbeConfig:
    n_prompts: int = 64
    max_new_tokens: int = 0          # 0 = teacher-forced on the prompt only, no generation
    context_tokens: int = 512        # truncate prompts to this many tokens
    train_frac: float = 0.8
    seed: int = 0
    layers: List[int] = field(default_factory=list)  # empty = all target layers
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


def load_prompts(rtype: str, n: int, seed: int) -> List[str]:
    from specloop_rt.real_corpus import build_corpus
    return build_corpus(rtype, n=n, seed=seed)


class LinearRoutingProbe(nn.Module):
    """One linear layer per target MoE decoder layer: draft hidden state
    (draft_hidden_size,) -> expert logits (num_experts,). Deliberately the
    simplest possible probe -- if a single linear map has no signal, a
    fancier probe head is very unlikely to rescue the pruning idea, and if
    it DOES have signal, that is exactly the minimum viable proxy the
    pruning method would want anyway (cheap enough to run in the draft
    step's critical path).
    """

    def __init__(self, draft_hidden_size: int, num_experts: int, num_target_layers: int):
        super().__init__()
        self.proj = nn.ModuleList(
            [nn.Linear(draft_hidden_size, num_experts) for _ in range(num_target_layers)]
        )

    def forward(self, draft_hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.proj[layer_idx](draft_hidden)


def topk_jaccard(pred_ids: torch.Tensor, true_ids: torch.Tensor) -> float:
    pred_set = set(pred_ids.tolist())
    true_set = set(true_ids.tolist())
    inter = len(pred_set & true_set)
    union = len(pred_set | true_set)
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--rtype", default="code", choices=["code", "rag", "chat", "reason"])
    ap.add_argument("--context-tokens", type=int, default=512)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--topk", type=int, default=8, help="target model's num_experts_per_tok")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_gpu_sweep/draft_routing_probe/result.json")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", HF_HOME_DEFAULT))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(args.hf_home, "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "0")

    torch.manual_seed(args.seed)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"[probe] loading tokenizer + target config from {TARGET_MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    target_config = AutoConfig.from_pretrained(TARGET_MODEL)
    num_experts = target_config.num_experts
    num_target_layers = target_config.num_hidden_layers
    aux_layer_ids = get_target_aux_layer_ids(target_config)
    print(f"[probe] target: {num_target_layers} layers, {num_experts} experts, "
          f"top-{target_config.num_experts_per_tok}; aux capture layers={aux_layer_ids}", flush=True)

    print(f"[probe] loading target model (bf16) -- this is the slow step", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        output_router_logits=True,
    )
    target.eval()

    print(f"[probe] loading draft head from {DRAFT_MODEL}", flush=True)
    draft_layer, draft_hidden_size = load_eagle3_draft_layer(DRAFT_MODEL, device="cuda", dtype=torch.bfloat16)
    # target's embedding table, reused by the draft head (no embed_tokens
    # tensor in the draft checkpoint -- confirmed in eagle3_draft_port.py).
    target_embed_tokens = target.get_input_embeddings()
    draft_load_mode = "standalone_port"

    print(f"[probe] loading {args.n_prompts} '{args.rtype}' prompts", flush=True)
    prompts = load_prompts(args.rtype, args.n_prompts, args.seed)

    all_draft_hidden: List[torch.Tensor] = []   # (n_tokens, draft_hidden_size) each
    all_true_topk: List[torch.Tensor] = []       # (n_tokens, num_target_layers, topk) each

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=args.context_tokens).input_ids.to("cuda")
            if ids.shape[1] < 8:
                continue

            out = target(ids, output_hidden_states=True, output_router_logits=True)
            # hidden_states: tuple of (num_layers+1) tensors, each (1, T, H)
            hs = out.hidden_states
            aux = torch.cat([hs[j + 1][0] for j in aux_layer_ids], dim=-1)  # (T, 3H)

            # router_logits: tuple of num_layers tensors, each (T, num_experts)
            # (batch=1 flattened by HF's Qwen3-MoE implementation)
            router_logits = out.router_logits  # tuple length num_target_layers
            true_topk_per_layer = []
            for layer_logits in router_logits:
                _, top_ids = torch.topk(layer_logits, k=args.topk, dim=-1)  # (T, topk)
                true_topk_per_layer.append(top_ids.cpu())
            true_topk = torch.stack(true_topk_per_layer, dim=1)  # (T, num_layers, topk)

            # EAGLE's own shift convention: the draft hidden state at
            # position t is built from embeds[t] (embedding of the CURRENT
            # token) and aux[t] (target's aux features at t), and is used
            # to PROPOSE the token at t+1 -- i.e. it is a prediction of
            # position t+1's identity/routing, not position t's own. Feed
            # embeds/aux at positions [0, T-2] and compare against true
            # routing at positions [1, T-1] (shift-by-one), matching what a
            # real draft step would be trying to anticipate before verify.
            positions = torch.arange(ids.shape[1], device="cuda")
            embeds = target_embed_tokens(ids)[0]  # (T, hidden_size)
            draft_hidden_all = draft_layer(embeds, aux, positions)  # (T, draft_hidden_size)

            draft_hidden = draft_hidden_all[:-1]         # predicts position t+1
            shifted_true_topk = true_topk[1:]             # ground truth AT t+1

            all_draft_hidden.append(draft_hidden.float().cpu())
            all_true_topk.append(shifted_true_topk)

            if (i + 1) % 10 == 0:
                print(f"[probe] {i+1}/{len(prompts)} prompts processed, "
                      f"{sum(t.shape[0] for t in all_draft_hidden)} tokens collected", flush=True)

    X = torch.cat(all_draft_hidden, dim=0)          # (N, draft_hidden_size)
    Y = torch.cat(all_true_topk, dim=0)              # (N, num_target_layers, topk)
    N = X.shape[0]
    print(f"[probe] collected {N} token positions total", flush=True)

    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    n_train = int(N * args.train_frac)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train, X_test = X[train_idx].cuda(), X[test_idx].cuda()
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    probe = LinearRoutingProbe(draft_hidden_size, num_experts, num_target_layers).cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)

    # multi-label BCE target: 1 for experts in the true top-k, 0 else
    def make_multilabel(topk_ids: torch.Tensor) -> torch.Tensor:
        # topk_ids: (n, num_target_layers, topk) -> (n, num_target_layers, num_experts)
        n = topk_ids.shape[0]
        target = torch.zeros(n, num_target_layers, num_experts)
        target.scatter_(2, topk_ids, 1.0)
        return target

    Y_train_ml = make_multilabel(Y_train).cuda()

    print(f"[probe] training linear probe: {n_train} train / {N - n_train} test tokens, "
          f"{num_target_layers} layers, {args.epochs} epochs", flush=True)
    for epoch in range(args.epochs):
        probe.train()
        opt.zero_grad()
        loss = 0.0
        for layer_idx in range(num_target_layers):
            logits = probe(X_train, layer_idx)
            loss = loss + F.binary_cross_entropy_with_logits(logits, Y_train_ml[:, layer_idx, :])
        loss = loss / num_target_layers
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[probe] epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}", flush=True)

    # --- evaluation: per-layer top-k Jaccard, probe vs baselines ---
    probe.eval()
    rng = torch.Generator().manual_seed(args.seed + 1)

    # frequency baseline: most common experts in the TRAIN split, per layer
    freq_topk_per_layer = []
    for layer_idx in range(num_target_layers):
        counts = torch.bincount(Y_train[:, layer_idx, :].reshape(-1), minlength=num_experts)
        freq_topk_per_layer.append(torch.topk(counts, k=args.topk).indices)

    results_per_layer = []
    with torch.no_grad():
        for layer_idx in range(num_target_layers):
            probe_logits = probe(X_test, layer_idx)
            probe_topk = torch.topk(probe_logits, k=args.topk, dim=-1).indices.cpu()

            n_test = X_test.shape[0]
            probe_scores, freq_scores, rand_scores = [], [], []
            for t in range(n_test):
                true_ids = Y_test[t, layer_idx, :]
                probe_scores.append(topk_jaccard(probe_topk[t], true_ids))
                freq_scores.append(topk_jaccard(freq_topk_per_layer[layer_idx], true_ids))
                rand_ids = torch.randperm(num_experts, generator=rng)[:args.topk]
                rand_scores.append(topk_jaccard(rand_ids, true_ids))

            results_per_layer.append({
                "layer": layer_idx,
                "probe_jaccard": sum(probe_scores) / n_test,
                "freq_baseline_jaccard": sum(freq_scores) / n_test,
                "random_baseline_jaccard": sum(rand_scores) / n_test,
            })

    mean_probe = sum(r["probe_jaccard"] for r in results_per_layer) / num_target_layers
    mean_freq = sum(r["freq_baseline_jaccard"] for r in results_per_layer) / num_target_layers
    mean_rand = sum(r["random_baseline_jaccard"] for r in results_per_layer) / num_target_layers

    print("\n" + "=" * 70)
    print(f"[probe] MEAN across {num_target_layers} target layers "
          f"(top-{args.topk} expert-set Jaccard overlap):")
    print(f"  probe (draft hidden -> linear head): {mean_probe:.4f}")
    print(f"  frequency baseline (train-set popular experts): {mean_freq:.4f}")
    print(f"  random baseline: {mean_rand:.4f}")
    print(f"  probe - freq margin: {mean_probe - mean_freq:+.4f}")
    print("=" * 70)
    if mean_probe > mean_freq + 0.03:
        print("[probe] VERDICT: probe beats frequency baseline by a real margin -- "
              "draft-side routing proxy has signal, footprint-aware pruning is worth building.")
    elif mean_probe > mean_freq:
        print("[probe] VERDICT: probe edges out frequency baseline but margin is small -- "
              "marginal signal, treat pruning payoff estimate as optimistic until GPU-validated.")
    else:
        print("[probe] VERDICT: probe does NOT beat the frequency baseline -- "
              "no evidence draft hidden state predicts target routing beyond global "
              "expert popularity. Footprint-aware pruning (as specified) has no signal "
              "to prune on; do not proceed to a live SGLang pruning hook without a "
              "different feature source.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": vars(args),
            "n_tokens_total": N,
            "n_train": n_train,
            "n_test": N - n_train,
            "mean_probe_jaccard": mean_probe,
            "mean_freq_baseline_jaccard": mean_freq,
            "mean_random_baseline_jaccard": mean_rand,
            "per_layer": results_per_layer,
            "draft_load_mode": draft_load_mode,
        }, f, indent=2)
    print(f"[probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
