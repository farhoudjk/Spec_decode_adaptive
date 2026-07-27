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


class _RealCorpusPool:
    """Lazily fetches and round-robins real prompts per rtype.

    Fetched once per rtype per process (datasets/ShareGPT downloads are
    seconds-to-minutes; every trace-generation call reuses the pool). Prompts
    repeat via modulo once exhausted -- acceptable for the trace lengths this
    harness runs (hundreds to low thousands of requests per rtype).
    """

    def __init__(self, seed: int = 0, pool_size: int = 2000):
        self.seed = seed
        self.pool_size = pool_size
        self._pools: Dict[str, List[str]] = {}

    def sample(self, rtype: str, rng: random.Random) -> str:
        if rtype not in self._pools:
            from . import real_corpus
            self._pools[rtype] = real_corpus.build_corpus(rtype, self.pool_size, seed=self.seed)
        pool = self._pools[rtype]
        return pool[rng.randrange(len(pool))]


def _draw_gap(nrng: np.random.Generator, rate: float, cv: float) -> float:
    """Inter-arrival gap with mean 1/rate and the given coefficient of variation.

    cv=1.0 is exponential (Poisson arrivals, the process's memoryless special
    case). cv!=1.0 uses a Gamma(shape=1/cv^2, scale=mean/shape) distribution,
    which has that mean and CV by construction; shape<1 (cv>1) concentrates
    mass near zero with a heavy tail -- i.e. many near-simultaneous arrivals
    separated by long quiet gaps, which is what "bursty" traffic means here.
    """
    mean = 1.0 / max(rate, 1e-9)
    if abs(cv - 1.0) < 1e-9:
        return float(nrng.exponential(mean))
    shape = 1.0 / (cv * cv)
    return float(nrng.gamma(shape, mean / shape))


def generate_trace(phases: List[tuple], seed: int = 0,
                   use_real_corpus: bool = False, cv: float = 1.0) -> List[TraceRequest]:
    """phases: list of (duration_s, {rtype: weight}, rate_rps).

    ``use_real_corpus=True`` draws prompts from specloop_rt.real_corpus
    (ShareGPT/HumanEval/SQuAD/CNN-DailyMail) instead of the synthetic
    templates below. Requires network access on first call per rtype
    (subsequent calls in the same process reuse the fetched pool).

    ``cv`` is the inter-arrival coefficient of variation: 1.0 (default)
    reproduces the original Poisson process bit-for-bit; >1.0 is burstier
    (see ``_draw_gap``). Applies uniformly across phases; use the
    ``bursty_*`` builders below for the paper's CV=2.5 arrival condition.
    """
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    reqs: List[TraceRequest] = []
    pool = _RealCorpusPool(seed=seed) if use_real_corpus else None
    t, rid = 0.0, 0
    for dur, mix, rate in phases:
        names = list(mix)
        p = np.array([mix[n] for n in names], float); p /= p.sum()
        t_end = t + dur
        while True:
            t += _draw_gap(nrng, rate, cv)
            if t >= t_end:
                t = t_end; break
            rt = REQUEST_TYPES[names[nrng.choice(len(names), p=p)]]
            olen = int(np.clip(nrng.lognormal(rt.output_mu, rt.output_sigma), 8, 1024))
            if pool is not None:
                prompt = pool.sample(rt.name, rng)
            else:
                nwords = max(8, int(nrng.normal(rt.prompt_words_mu, 0.2 * rt.prompt_words_mu)))
                prompt = _TEMPLATES[rt.name].format(ctx=_corpus_ctx(rt.name, rng, nwords))
            reqs.append(TraceRequest(f"r{rid}", t, prompt, olen, rt.name))
            rid += 1
    reqs.sort(key=lambda r: r.arrival_s)
    return reqs


# ---- canned traces mirroring the simulator study -------------------------
# Poisson (cv=1.0) arrivals, at whatever rate_rps is passed in.

def step_perturbation(rate=8.0, warmup=60.0, post=120.0, seed=0, use_real_corpus=False):
    return generate_trace([(warmup, {"rag": 0.6, "code": 0.4}, rate),
                           (post, {"chat": 0.6, "reason": 0.4}, rate)], seed,
                          use_real_corpus=use_real_corpus)

def mixed(rate=8.0, duration=180.0, seed=0, use_real_corpus=False):
    return generate_trace([(duration, {"rag": 0.3, "code": 0.2, "chat": 0.35, "reason": 0.15}, rate)], seed,
                          use_real_corpus=use_real_corpus)

def volatile(rate=8.0, duration=180.0, switch=30.0, seed=0, use_real_corpus=False):
    phases, t, i = [], 0.0, 0
    hi, lo = {"rag": 0.6, "code": 0.4}, {"chat": 0.6, "reason": 0.4}
    while t < duration:
        d = min(switch, duration - t)
        phases.append((d, hi if i % 2 == 0 else lo, rate)); t += d; i += 1
    return generate_trace(phases, seed, use_real_corpus=use_real_corpus)

def homogeneous(rtype="chat", rate=8.0, duration=120.0, seed=0, use_real_corpus=False):
    return generate_trace([(duration, {rtype: 1.0}, rate)], seed, use_real_corpus=use_real_corpus)


# ---- bursty variants: same mixes/phases, CV=2.5 inter-arrival gaps -------
# Same mean rate as the Poisson builders above, but arrivals cluster into
# bursts separated by longer quiet periods (Gamma(shape=1/cv^2) gaps). This
# is the "bursty" arrival condition the paper's rate-sweep asks for,
# alongside Poisson at various rates.

BURSTY_CV = 2.5

def bursty_step_perturbation(rate=8.0, warmup=60.0, post=120.0, seed=0,
                             use_real_corpus=False, cv=BURSTY_CV):
    return generate_trace([(warmup, {"rag": 0.6, "code": 0.4}, rate),
                           (post, {"chat": 0.6, "reason": 0.4}, rate)], seed,
                          use_real_corpus=use_real_corpus, cv=cv)

def bursty_mixed(rate=8.0, duration=180.0, seed=0, use_real_corpus=False, cv=BURSTY_CV):
    return generate_trace([(duration, {"rag": 0.3, "code": 0.2, "chat": 0.35, "reason": 0.15}, rate)], seed,
                          use_real_corpus=use_real_corpus, cv=cv)

def bursty_volatile(rate=8.0, duration=180.0, switch=30.0, seed=0,
                    use_real_corpus=False, cv=BURSTY_CV):
    phases, t, i = [], 0.0, 0
    hi, lo = {"rag": 0.6, "code": 0.4}, {"chat": 0.6, "reason": 0.4}
    while t < duration:
        d = min(switch, duration - t)
        phases.append((d, hi if i % 2 == 0 else lo, rate)); t += d; i += 1
    return generate_trace(phases, seed, use_real_corpus=use_real_corpus, cv=cv)

def bursty_homogeneous(rtype="chat", rate=8.0, duration=120.0, seed=0,
                       use_real_corpus=False, cv=BURSTY_CV):
    return generate_trace([(duration, {rtype: 1.0}, rate)], seed,
                          use_real_corpus=use_real_corpus, cv=cv)


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
