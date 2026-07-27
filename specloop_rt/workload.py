"""Workload trace generation and replay driver for the real engine.

Traces are generated the same way as the simulator study (so results are
comparable) but are consumed by an async client that submits real prompts to
the vLLM AsyncLLMEngine.  Prompts are drawn from a corpus to hit realistic
acceptance behaviour; the trace fixes ARRIVAL TIME, PROMPT, and MAX_TOKENS per
request, which is what makes runs across controllers reproducible.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TraceRequest:
    rid: str
    arrival_s: float
    prompt: str
    max_tokens: int
    rtype: str


# Prompt templates by acceptance regime. Repetitive/structured -> high draft
# acceptance; open-ended -> low.  The corpus is intentionally simple; swap in
# ShareGPT/HumanEval/RAG prompts for camera-ready.
_TEMPLATES = {
    "rag": "Repeat and lightly summarize the following passage, preserving wording:\n{ctx}\n\nSummary:",
    "code": "Complete this function in the same style:\n{ctx}\n",
    "chat": "{ctx}",
    "reason": "Solve step by step, showing all work:\n{ctx}\n",
}


def _corpus_ctx(rtype: str, rng: random.Random, n_words: int) -> str:
    banks = {
        "rag": "the quarterly report indicates revenue growth across all regions ",
        "code": "def process(items):\n    total = 0\n    for it in items:\n        total += it.value\n    ",
        "chat": "what do you think about the future of renewable energy and its impact ",
        "reason": "if a train travels 60 km in 45 minutes and then 30 km in 20 minutes ",
    }
    base = banks[rtype]
    reps = max(1, n_words // max(1, len(base.split())))
    return (base * reps).strip()


@dataclass
class RequestType:
    name: str
    prompt_words_mu: float
    output_mu: float
    output_sigma: float
    weight: float = 1.0


REQUEST_TYPES = {
    "rag":    RequestType("rag", 220, 4.6, 0.6),
    "code":   RequestType("code", 160, 5.3, 0.7),
    "chat":   RequestType("chat", 80, 6.0, 0.8),
    "reason": RequestType("reason", 120, 7.0, 0.7),
}


def generate_trace(phases: List[tuple], seed: int = 0) -> List[TraceRequest]:
    """phases: list of (duration_s, {rtype: weight}, rate_rps)."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    reqs: List[TraceRequest] = []
    t, rid = 0.0, 0
    for dur, mix, rate in phases:
        names = list(mix)
        p = np.array([mix[n] for n in names], float); p /= p.sum()
        t_end = t + dur
        while True:
            t += nrng.exponential(1.0 / max(rate, 1e-9))
            if t >= t_end:
                t = t_end; break
            rt = REQUEST_TYPES[names[nrng.choice(len(names), p=p)]]
            nwords = max(8, int(nrng.normal(rt.prompt_words_mu, 0.2 * rt.prompt_words_mu)))
            olen = int(np.clip(nrng.lognormal(rt.output_mu, rt.output_sigma), 8, 1024))
            prompt = _TEMPLATES[rt.name].format(ctx=_corpus_ctx(rt.name, rng, nwords))
            reqs.append(TraceRequest(f"r{rid}", t, prompt, olen, rt.name))
            rid += 1
    reqs.sort(key=lambda r: r.arrival_s)
    return reqs


# ---- canned traces mirroring the simulator study -------------------------

def step_perturbation(rate=8.0, warmup=60.0, post=120.0, seed=0):
    return generate_trace([(warmup, {"rag": 0.6, "code": 0.4}, rate),
                           (post, {"chat": 0.6, "reason": 0.4}, rate)], seed)

def mixed(rate=8.0, duration=180.0, seed=0):
    return generate_trace([(duration, {"rag": 0.3, "code": 0.2, "chat": 0.35, "reason": 0.15}, rate)], seed)

def volatile(rate=8.0, duration=180.0, switch=30.0, seed=0):
    phases, t, i = [], 0.0, 0
    hi, lo = {"rag": 0.6, "code": 0.4}, {"chat": 0.6, "reason": 0.4}
    while t < duration:
        d = min(switch, duration - t)
        phases.append((d, hi if i % 2 == 0 else lo, rate)); t += d; i += 1
    return generate_trace(phases, seed)

def homogeneous(rtype="chat", rate=8.0, duration=120.0, seed=0):
    return generate_trace([(duration, {rtype: 1.0}, rate)], seed)


def save_trace(reqs: List[TraceRequest], path: str):
    with open(path, "w") as f:
        for r in reqs:
            f.write(json.dumps(r.__dict__) + "\n")

def load_trace(path: str) -> List[TraceRequest]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(TraceRequest(**json.loads(line)))
    return out
