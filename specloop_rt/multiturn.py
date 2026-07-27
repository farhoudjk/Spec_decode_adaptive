"""Multi-turn conversation trace + replay driver.

Single-shot TraceRequest (workload.py) fires independent, pre-scheduled
requests: arrival time, prompt, and max_tokens are all fixed up front, so the
whole trace can be built once and replayed by racing N independent tasks.
That model breaks for a real multi-turn conversation: turn k+1's prompt is
turn 1..k's human text PLUS the model's own turn 1..k replies, so it cannot be
known before turn k actually finishes generating, and it cannot be submitted
before turn k finishes plus some human "think time". This module is the
sequential-per-conversation counterpart: one asyncio task per conversation,
looping turn -> wait for completion -> think-time gap -> append history -> next
turn, with the growing shared-prefix prompt handed to vLLM each time so
automatic prefix caching (must be enabled on the engine) reuses the KV blocks
already computed for the conversation's earlier turns.

Docs: `AsyncEngineArgs.enable_prefix_caching` (vllm.engine.arg_utils) --
enabled by default via configs/*.yaml overrides in replay.py's _build_engine
callers that opt into multi-turn.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import yaml

from specloop_rt import real_corpus as RC
from specloop_rt.controllers import build_controller
from specloop_rt.interface import TelemetryWriter

# ---------------------------------------------------------------------------
# Trace construction
# ---------------------------------------------------------------------------


@dataclass
class ConversationTrace:
    cid: str
    arrival_s: float          # when the FIRST turn is submitted
    turns: List[str]          # human message per turn, in order
    max_tokens_per_turn: List[int]


@dataclass
class TurnResult:
    cid: str
    turn_idx: int
    request_id: str
    submit_wall: float
    first_token_wall: Optional[float] = None
    finish_wall: Optional[float] = None
    output_tokens: int = 0
    prompt_tokens_est: int = 0   # len(prompt) chars, cheap proxy logged for prefix-reuse sanity checks
    ttft_s: Optional[float] = None
    tpot_s: Optional[float] = None
    e2e_s: Optional[float] = None
    output_text: str = ""


def build_multiturn_trace(n_conversations: int, rate: float, seed: int = 0,
                          min_turns: int = 2, max_turns: int = 6,
                          output_mu: float = 5.5, output_sigma: float = 0.7) -> List[ConversationTrace]:
    """n_conversations first-turn arrivals as a Poisson process at `rate`,
    each drawing a real multi-turn ShareGPT conversation (min_turns..max_turns
    human messages; conversations shorter than min_turns are excluded, longer
    ones truncated to max_turns so no single conversation dominates run time).
    """
    convs = RC.load_sharegpt_conversations(min_turns=min_turns, seed=seed)
    if not convs:
        raise RuntimeError("no ShareGPT conversations with >= min_turns human turns")

    nrng = np.random.default_rng(seed)
    out: List[ConversationTrace] = []
    t = 0.0
    for i in range(n_conversations):
        t += float(nrng.exponential(1.0 / max(rate, 1e-9)))
        conv = convs[i % len(convs)]
        turns = conv.turns[:max_turns]
        max_toks = [int(np.clip(nrng.lognormal(output_mu, output_sigma), 8, 1024))
                    for _ in turns]
        out.append(ConversationTrace(cid=f"conv{i}_{conv.cid}", arrival_s=t,
                                     turns=turns, max_tokens_per_turn=max_toks))
    return out


def save_trace(convs: List[ConversationTrace], path: str) -> None:
    with open(path, "w") as f:
        for c in convs:
            f.write(json.dumps(asdict(c)) + "\n")


def load_trace(path: str) -> List[ConversationTrace]:
    out = []
    with open(path) as f:
        for line in f:
            out.append(ConversationTrace(**json.loads(line)))
    return out


# ---------------------------------------------------------------------------
# Prompt assembly: turn k's prompt is the full conversation so far
# ---------------------------------------------------------------------------


def _render_prompt(history: List[str]) -> str:
    """history alternates human/assistant text, human first. Simple chat
    template -- swap for the target model's real chat template if tokenizer
    mismatches matter for your acceptance numbers (see README caveat)."""
    lines = []
    for i, msg in enumerate(history):
        role = "User" if i % 2 == 0 else "Assistant"
        lines.append(f"{role}: {msg}")
    lines.append("Assistant:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


async def _run_conversation(engine, conv: ConversationTrace, sampling, t0: float,
                            think_time_mu: float, think_time_sigma: float,
                            rng: np.random.Generator,
                            results: List[TurnResult]) -> None:
    from vllm import SamplingParams  # noqa

    dt = conv.arrival_s - (time.monotonic() - t0)
    if dt > 0:
        await asyncio.sleep(dt)

    history: List[str] = []
    for k, (human_msg, max_tok) in enumerate(zip(conv.turns, conv.max_tokens_per_turn)):
        history.append(human_msg)
        prompt = _render_prompt(history)
        rid = f"{conv.cid}_t{k}"
        submit = time.monotonic()
        tr = TurnResult(cid=conv.cid, turn_idx=k, request_id=rid, submit_wall=submit,
                        prompt_tokens_est=len(prompt))
        sp = sampling(max_tok)
        n = 0
        out_text = ""
        async for out in engine.generate(prompt, sp, request_id=rid):
            now = time.monotonic()
            toks = len(out.outputs[0].token_ids) if out.outputs else 0
            if tr.first_token_wall is None and toks > 0:
                tr.first_token_wall = now
                tr.ttft_s = now - submit
            n = toks
            if out.finished:
                tr.finish_wall = now
                tr.output_tokens = n
                tr.e2e_s = now - submit
                out_text = out.outputs[0].text if out.outputs else ""
                if n > 0 and tr.first_token_wall is not None:
                    tr.tpot_s = (now - tr.first_token_wall) / max(1, n - 1)
                break
        tr.output_text = out_text
        results.append(tr)
        history.append(out_text)

        if k < len(conv.turns) - 1:
            think = float(rng.lognormal(think_time_mu, think_time_sigma))
            await asyncio.sleep(think)


async def replay_multiturn(cfg: dict, convs: List[ConversationTrace], out_dir: str,
                           think_time_mu: float = 1.6, think_time_sigma: float = 0.8) -> List[TurnResult]:
    """think_time_mu/sigma parametrize a lognormal in seconds; mu=1.6,
    sigma=0.8 gives a median think-time of ~5s with a long right tail, in
    line with human multi-turn chat pacing figures used elsewhere in the
    serving-benchmark literature. Independent of arrival rate/cv -- those
    control only how fast NEW conversations start, not pacing within one.
    """
    from specloop_rt.replay import _build_engine  # reuse the same engine builder

    os.makedirs(out_dir, exist_ok=True)
    controller = build_controller(cfg["controller"], cfg["runtime"]["tpot_slo_s"])
    telemetry = TelemetryWriter(os.path.join(out_dir, "steps.jsonl"),
                                run_meta={"cfg": cfg, "n_conversations": len(convs),
                                          "mode": "multiturn"})

    from specloop_rt.vllm_patch import configure
    configure(controller, telemetry, cfg["runtime"])

    engine = _build_engine(cfg)

    def sampling(mt):
        from vllm import SamplingParams
        return SamplingParams(temperature=cfg["runtime"].get("temperature", 0.0),
                              max_tokens=mt, ignore_eos=cfg["runtime"].get("ignore_eos", False))

    rng = np.random.default_rng(cfg["runtime"].get("seed", 0))
    results: List[TurnResult] = []
    t0 = time.monotonic()
    tasks = [asyncio.create_task(_run_conversation(engine, c, sampling, t0,
                                                   think_time_mu, think_time_sigma, rng, results))
             for c in convs]
    await asyncio.gather(*tasks)
    telemetry.close()

    with open(os.path.join(out_dir, "requests.jsonl"), "w") as f:
        for tr in results:
            f.write(json.dumps(asdict(tr)) + "\n")
    return results


def main(argv=None):
    p = argparse.ArgumentParser("specloop-replay-multiturn")
    p.add_argument("--config", required=True)
    p.add_argument("--n-conversations", type=int, default=100)
    p.add_argument("--rate", type=float, default=1.0,
                   help="new-conversation arrival rate (conversations/sec), Poisson")
    p.add_argument("--min-turns", type=int, default=2)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--think-time-mu", type=float, default=1.6)
    p.add_argument("--think-time-sigma", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results_gpu/multiturn_run")
    a = p.parse_args(argv)
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    cfg["runtime"]["seed"] = a.seed

    convs = build_multiturn_trace(a.n_conversations, a.rate, seed=a.seed,
                                  min_turns=a.min_turns, max_turns=a.max_turns)
    if not os.path.isdir(a.out):
        save_trace(convs, a.out + "_trace.jsonl")
    asyncio.run(replay_multiturn(cfg, convs, a.out,
                                 think_time_mu=a.think_time_mu,
                                 think_time_sigma=a.think_time_sigma))
    n_turns = sum(len(c.turns) for c in convs)
    print(f"done: {len(convs)} conversations, {n_turns} turns -> {a.out}")


if __name__ == "__main__":
    main()
