"""Real-dataset prompt sources: ShareGPT, HumanEval, SQuAD, CNN-DailyMail.

Replaces workload.py's synthetic templates with actual prompts, so acceptance
behaviour (draft/target agreement) comes from real text rather than a
hand-picked alpha distribution. This module has no vLLM import; it only
depends on ``datasets``/``huggingface_hub`` and is safe to run off-GPU.

Dataset -> rtype mapping (mirrors workload.py's REQUEST_TYPES so downstream
code that groups by rtype keeps working):
    rag    -> SQuAD          (long context + short extractive answer)
    code   -> HumanEval       (function-completion prompt)
    chat   -> ShareGPT        (first human turn of a conversation)
    reason -> CNN-DailyMail   (long-form summarization, treated as the
                               "long output, decaying acceptance" analogue)

ShareGPT conversations are naturally multi-turn (alternating human/gpt turns).
``load_sharegpt_conversations`` exposes the full turn sequence for the
multi-turn trace generator; ``sharegpt_first_turns`` collapses each
conversation to its opening prompt for the single-turn corpora below.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

CACHE_DIR = os.environ.get("SPECLOOP_HF_CACHE", os.path.expanduser("~/.cache/specloop_rt"))

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"


# ---------------------------------------------------------------------------
# Multi-turn conversation representation (ShareGPT)
# ---------------------------------------------------------------------------


@dataclass
class Conversation:
    cid: str
    turns: List[str]           # human turns only, in order


def _download_sharegpt_json() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=SHAREGPT_REPO, filename=SHAREGPT_FILE,
                           repo_type="dataset", cache_dir=CACHE_DIR)


def load_sharegpt_conversations(min_turns: int = 1, max_conversations: Optional[int] = None,
                                seed: int = 0) -> List[Conversation]:
    """Full multi-turn ShareGPT conversations (human turns only, gpt turns dropped
    -- the replay client generates its own responses; only the human side of the
    recorded conversation is replayed as trace input).
    """
    path = _download_sharegpt_json()
    with open(path) as f:
        raw = json.load(f)

    convs: List[Conversation] = []
    for item in raw:
        human_turns = [c["value"] for c in item.get("conversations", [])
                       if c.get("from") == "human" and c.get("value", "").strip()]
        if len(human_turns) >= min_turns:
            convs.append(Conversation(cid=item["id"], turns=human_turns))

    rng = random.Random(seed)
    rng.shuffle(convs)
    if max_conversations is not None:
        convs = convs[:max_conversations]
    return convs


def sharegpt_first_turns(n: int, seed: int = 0) -> List[str]:
    """First human message of n distinct conversations, for single-turn use."""
    convs = load_sharegpt_conversations(min_turns=1, max_conversations=None, seed=seed)
    return [c.turns[0] for c in convs[:n]]


# ---------------------------------------------------------------------------
# Single-turn corpora
# ---------------------------------------------------------------------------


def humaneval_prompts(n: Optional[int] = None) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test", cache_dir=CACHE_DIR)
    prompts = [f"Complete this Python function:\n\n{ex['prompt']}" for ex in ds]
    return prompts[:n] if n else prompts


def squad_prompts(n: Optional[int] = None, seed: int = 0) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation", cache_dir=CACHE_DIR)
    ds = ds.shuffle(seed=seed)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return [f"Context: {ex['context']}\n\nQuestion: {ex['question']}\n\nAnswer:"
            for ex in ds]


def cnn_dailymail_prompts(n: Optional[int] = None, seed: int = 0) -> List[str]:
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", cache_dir=CACHE_DIR)
    ds = ds.shuffle(seed=seed)
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return [f"Summarize the following article:\n\n{ex['article']}\n\nSummary:"
            for ex in ds]


# ---------------------------------------------------------------------------
# rtype -> corpus dispatch, matching workload.py's REQUEST_TYPES names
# ---------------------------------------------------------------------------

CORPUS_BUILDERS = {
    "rag": squad_prompts,
    "code": humaneval_prompts,
    "chat": lambda n=None, seed=0: sharegpt_first_turns(n or 2000, seed=seed),
    "reason": cnn_dailymail_prompts,
}


def build_corpus(rtype: str, n: int, seed: int = 0) -> List[str]:
    fn = CORPUS_BUILDERS[rtype]
    try:
        return fn(n=n, seed=seed)
    except TypeError:
        return fn(n)
