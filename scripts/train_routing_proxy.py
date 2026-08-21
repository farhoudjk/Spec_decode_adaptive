"""Train the real RoutingProxyHead checkpoint for footprint-aware draft
pruning (idea 1), superseding the diagnostic probe
(probe_draft_routing_predictability.py) which validated the underlying
signal exists (mean Jaccard 0.52 vs 0.22 frequency baseline, all 48/48
target layers, code-only, 64 prompts -- see
results_gpu_sweep/draft_routing_probe/result.json) but was never meant to
ship: single workload, throwaway weights, no saved checkpoint.

This script:
  1. Pulls prompts from ALL FOUR workloads this repo already has corpus
     builders for (code/HumanEval, rag/SQuAD, chat/ShareGPT,
     reason/CNN-DailyMail) -- matching the same workload taxonomy Axis-7's
     own code/reason generalization check used, not inventing a new one.
  2. Extracts (draft hidden state, true target routing) pairs with the
     shared Eagle3DraftLayer port (specloop_rt/sglang_patch/
     eagle3_draft_port.py) -- same feature extraction as the probe,
     factored out so the two scripts can't silently drift apart.
  3. Trains specloop_rt.sglang_patch.footprint_aware_pruning.RoutingProxyHead
     (the actual class the pruning hook imports, not a throwaway local
     copy) and saves it as a checkpoint.
  4. Evaluates on TWO held-out splits, not one:
       (a) random held-out tokens, mixed across all workloads trained on
           (in-distribution check -- same thing the probe measured)
       (b) an entire workload withheld from training and used only for
           eval (cross-workload generalization check -- the probe never
           tested this, and a proxy that only works on the one workload
           it saw isn't a usable serving-time artifact)

Position-shift convention (load-bearing, identical to the probe): draft
hidden state at position t predicts target routing at t+1, not at t. See
probe_draft_routing_predictability.py's docstring for why getting this
backwards would inflate the result meaninglessly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)

from specloop_rt.sglang_patch.eagle3_draft_port import (
    get_target_aux_layer_ids,
    load_eagle3_draft_layer,
)
from specloop_rt.sglang_patch.footprint_aware_pruning import RoutingProxyHead

TARGET_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DRAFT_MODEL = "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex"
HF_HOME_DEFAULT = "/root/hf_cache"

ALL_WORKLOADS = ["code", "rag", "chat", "reason"]


def topk_jaccard(pred_ids: torch.Tensor, true_ids: torch.Tensor) -> float:
    pred_set = set(pred_ids.tolist())
    true_set = set(true_ids.tolist())
    inter = len(pred_set & true_set)
    union = len(pred_set | true_set)
    return inter / union if union else 0.0


def extract_features(
    target, draft_layer, tok, target_embed_tokens, aux_layer_ids: List[int],
    rtype: str, n_prompts: int, context_tokens: int, topk: int, seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (X, Y): X is (N, draft_hidden_size) draft hidden states, Y is
    (N, num_target_layers, topk) true expert ids, both already
    position-shifted (X[i] predicts Y[i], i.e. Y is at t+1 relative to X's
    t) and pooled across all prompts of one workload."""
    from specloop_rt.real_corpus import build_corpus
    prompts = build_corpus(rtype, n=n_prompts, seed=seed)

    all_draft_hidden: List[torch.Tensor] = []
    all_true_topk: List[torch.Tensor] = []

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=context_tokens).input_ids.to("cuda")
            if ids.shape[1] < 8:
                continue

            out = target(ids, output_hidden_states=True, output_router_logits=True)
            hs = out.hidden_states
            aux = torch.cat([hs[j + 1][0] for j in aux_layer_ids], dim=-1)

            router_logits = out.router_logits
            true_topk_per_layer = [
                torch.topk(layer_logits, k=topk, dim=-1).indices.cpu()
                for layer_logits in router_logits
            ]
            true_topk = torch.stack(true_topk_per_layer, dim=1)

            positions = torch.arange(ids.shape[1], device="cuda")
            embeds = target_embed_tokens(ids)[0]
            draft_hidden_all = draft_layer(embeds, aux, positions)

            all_draft_hidden.append(draft_hidden_all[:-1].float().cpu())
            all_true_topk.append(true_topk[1:])

            if (i + 1) % 25 == 0:
                print(f"[train] [{rtype}] {i+1}/{len(prompts)} prompts, "
                      f"{sum(t.shape[0] for t in all_draft_hidden)} tokens so far", flush=True)

    X = torch.cat(all_draft_hidden, dim=0)
    Y = torch.cat(all_true_topk, dim=0)
    print(f"[train] [{rtype}] done: {X.shape[0]} token positions", flush=True)
    return X, Y


def evaluate(probe: RoutingProxyHead, X: torch.Tensor, Y: torch.Tensor,
            freq_topk_per_layer: List[torch.Tensor], num_target_layers: int,
            num_experts: int, topk: int, seed: int) -> Dict:
    probe.eval()
    rng = torch.Generator().manual_seed(seed)
    X = X.cuda()
    results_per_layer = []
    with torch.no_grad():
        for layer_idx in range(num_target_layers):
            logits = probe.proj[layer_idx](X)
            pred_topk = torch.topk(logits, k=topk, dim=-1).indices.cpu()
            n = X.shape[0]
            probe_scores, freq_scores, rand_scores = [], [], []
            for t in range(n):
                true_ids = Y[t, layer_idx, :]
                probe_scores.append(topk_jaccard(pred_topk[t], true_ids))
                freq_scores.append(topk_jaccard(freq_topk_per_layer[layer_idx], true_ids))
                rand_ids = torch.randperm(num_experts, generator=rng)[:topk]
                rand_scores.append(topk_jaccard(rand_ids, true_ids))
            results_per_layer.append({
                "layer": layer_idx,
                "probe_jaccard": sum(probe_scores) / n,
                "freq_baseline_jaccard": sum(freq_scores) / n,
                "random_baseline_jaccard": sum(rand_scores) / n,
            })
    mean_probe = sum(r["probe_jaccard"] for r in results_per_layer) / num_target_layers
    mean_freq = sum(r["freq_baseline_jaccard"] for r in results_per_layer) / num_target_layers
    mean_rand = sum(r["random_baseline_jaccard"] for r in results_per_layer) / num_target_layers
    return {
        "n_tokens": X.shape[0],
        "mean_probe_jaccard": mean_probe,
        "mean_freq_baseline_jaccard": mean_freq,
        "mean_random_baseline_jaccard": mean_rand,
        "per_layer": results_per_layer,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-prompts-per-workload", type=int, default=150)
    ap.add_argument("--holdout-workload", default="reason", choices=ALL_WORKLOADS,
                    help="Entirely excluded from training; used only for the "
                         "cross-workload generalization eval (not the random "
                         "in-distribution split).")
    ap.add_argument("--context-tokens", type=int, default=512)
    ap.add_argument("--train-frac", type=float, default=0.85,
                    help="Random split fraction WITHIN the non-holdout workloads' "
                         "pooled tokens (in-distribution eval uses the remainder).")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--topk", type=int, default=8, help="target model's num_experts_per_tok")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results_gpu_sweep/routing_proxy_train")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", HF_HOME_DEFAULT))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(args.hf_home, "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "0")

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print(f"[train] loading tokenizer + target config from {TARGET_MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    target_config = AutoConfig.from_pretrained(TARGET_MODEL)
    num_experts = target_config.num_experts
    num_target_layers = target_config.num_hidden_layers
    aux_layer_ids = get_target_aux_layer_ids(target_config)
    print(f"[train] target: {num_target_layers} layers, {num_experts} experts, "
          f"top-{target_config.num_experts_per_tok}; aux capture layers={aux_layer_ids}", flush=True)

    print("[train] loading target model (bf16)", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        output_router_logits=True,
    )
    target.eval()

    print(f"[train] loading draft head from {DRAFT_MODEL}", flush=True)
    draft_layer, draft_hidden_size = load_eagle3_draft_layer(DRAFT_MODEL, device="cuda", dtype=torch.bfloat16)
    target_embed_tokens = target.get_input_embeddings()

    train_workloads = [w for w in ALL_WORKLOADS if w != args.holdout_workload]
    print(f"[train] training workloads: {train_workloads}; held out entirely: {args.holdout_workload}", flush=True)

    X_by_workload: Dict[str, torch.Tensor] = {}
    Y_by_workload: Dict[str, torch.Tensor] = {}
    for rtype in ALL_WORKLOADS:  # extract holdout too, for its own eval
        X, Y = extract_features(
            target, draft_layer, tok, target_embed_tokens, aux_layer_ids,
            rtype, args.n_prompts_per_workload, args.context_tokens, args.topk, args.seed,
        )
        X_by_workload[rtype] = X
        Y_by_workload[rtype] = Y

    X_train_pool = torch.cat([X_by_workload[w] for w in train_workloads], dim=0)
    Y_train_pool = torch.cat([Y_by_workload[w] for w in train_workloads], dim=0)
    N = X_train_pool.shape[0]
    print(f"[train] pooled training-workload tokens: {N}", flush=True)

    perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed))
    n_train = int(N * args.train_frac)
    train_idx, indist_test_idx = perm[:n_train], perm[n_train:]

    X_train, X_indist_test = X_train_pool[train_idx].cuda(), X_train_pool[indist_test_idx]
    Y_train, Y_indist_test = Y_train_pool[train_idx], Y_train_pool[indist_test_idx]

    probe = RoutingProxyHead(draft_hidden_size, num_experts, num_target_layers).cuda()
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def make_multilabel(topk_ids: torch.Tensor) -> torch.Tensor:
        n = topk_ids.shape[0]
        t = torch.zeros(n, num_target_layers, num_experts)
        t.scatter_(2, topk_ids, 1.0)
        return t

    Y_train_ml = make_multilabel(Y_train).cuda()

    print(f"[train] training: {n_train} train / {N - n_train} in-dist test tokens, "
          f"{num_target_layers} layers, {args.epochs} epochs", flush=True)
    for epoch in range(args.epochs):
        probe.train()
        opt.zero_grad()
        loss = 0.0
        for layer_idx in range(num_target_layers):
            logits = probe.proj[layer_idx](X_train)
            loss = loss + F.binary_cross_entropy_with_logits(logits, Y_train_ml[:, layer_idx, :])
        loss = loss / num_target_layers
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[train] epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}", flush=True)

    freq_topk_per_layer = [
        torch.topk(torch.bincount(Y_train[:, l, :].reshape(-1), minlength=num_experts), k=args.topk).indices
        for l in range(num_target_layers)
    ]

    print("\n" + "=" * 70, flush=True)
    print("[train] IN-DISTRIBUTION eval (held-out tokens, same workloads as training)", flush=True)
    indist_result = evaluate(probe, X_indist_test, Y_indist_test, freq_topk_per_layer,
                             num_target_layers, num_experts, args.topk, args.seed + 1)
    print(f"  n_tokens={indist_result['n_tokens']}  probe={indist_result['mean_probe_jaccard']:.4f}  "
          f"freq={indist_result['mean_freq_baseline_jaccard']:.4f}  "
          f"margin={indist_result['mean_probe_jaccard']-indist_result['mean_freq_baseline_jaccard']:+.4f}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"[train] CROSS-WORKLOAD eval (held-out workload: '{args.holdout_workload}', "
          f"never seen in training)", flush=True)
    holdout_result = evaluate(probe, X_by_workload[args.holdout_workload], Y_by_workload[args.holdout_workload],
                              freq_topk_per_layer, num_target_layers, num_experts, args.topk, args.seed + 2)
    print(f"  n_tokens={holdout_result['n_tokens']}  probe={holdout_result['mean_probe_jaccard']:.4f}  "
          f"freq={holdout_result['mean_freq_baseline_jaccard']:.4f}  "
          f"margin={holdout_result['mean_probe_jaccard']-holdout_result['mean_freq_baseline_jaccard']:+.4f}", flush=True)
    print("=" * 70, flush=True)

    gap = indist_result["mean_probe_jaccard"] - holdout_result["mean_probe_jaccard"]
    print(f"\n[train] Generalization gap (in-dist probe - cross-workload probe): {gap:+.4f}", flush=True)
    if holdout_result["mean_probe_jaccard"] > holdout_result["mean_freq_baseline_jaccard"] + 0.03:
        print(f"[train] VERDICT: proxy generalizes to an unseen workload ('{args.holdout_workload}') "
              f"with a real margin over frequency baseline. Safe to ship as a general-purpose proxy.", flush=True)
    elif holdout_result["mean_probe_jaccard"] > holdout_result["mean_freq_baseline_jaccard"]:
        print(f"[train] VERDICT: proxy edges out frequency baseline on '{args.holdout_workload}' but margin "
              f"is small -- treat as workload-specific until more holdout workloads are checked.", flush=True)
    else:
        print(f"[train] VERDICT: proxy does NOT beat frequency baseline on unseen workload "
              f"'{args.holdout_workload}' -- signal may be workload-specific (e.g. overfit to code-like "
              f"token distributions), not a general routing predictor. Re-check before shipping.", flush=True)

    ckpt_path = os.path.join(args.out_dir, "routing_proxy_head.pt")
    torch.save({
        "state_dict": probe.state_dict(),
        "draft_hidden_size": draft_hidden_size,
        "num_experts": num_experts,
        "num_target_layers": num_target_layers,
        "topk": args.topk,
        "train_workloads": train_workloads,
        "holdout_workload": args.holdout_workload,
        "target_model": TARGET_MODEL,
        "draft_model": DRAFT_MODEL,
    }, ckpt_path)
    print(f"\n[train] saved checkpoint to {ckpt_path}", flush=True)

    result_path = os.path.join(args.out_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump({
            "config": vars(args),
            "n_train": n_train,
            "in_distribution": indist_result,
            "cross_workload_holdout": holdout_result,
            "generalization_gap": gap,
            "checkpoint_path": ckpt_path,
        }, f, indent=2)
    print(f"[train] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
