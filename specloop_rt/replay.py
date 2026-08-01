"""Async trace-replay driver against vLLM's AsyncLLMEngine.

The engine runs with our SpecLoopScheduler (per-step control + telemetry). This
client is responsible only for the OPEN-LOOP arrival process: it sleeps until
each request's scheduled arrival time and submits it, then records per-request
TTFT/TPOT/E2E from the streaming output. Per-request metrics + the scheduler's
per-step telemetry together give every number the paper needs.

vLLM import is confined to ``_build_engine`` so the module imports on a machine
without vllm (for trace prep / dry runs).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import yaml

from specloop_rt import workload as W
from specloop_rt.controllers import build_controller
from specloop_rt.interface import TelemetryWriter


@dataclass
class RequestResult:
    rid: str
    rtype: str
    arrival_s: float
    submit_wall: float
    first_token_wall: Optional[float] = None
    finish_wall: Optional[float] = None
    output_tokens: int = 0
    ttft_s: Optional[float] = None
    tpot_s: Optional[float] = None
    e2e_s: Optional[float] = None
    # Load shedding: a rejected request never reaches the engine, so it has no
    # TTFT/TPOT/e2e. It is NOT a vanished request -- analysis counts it as an
    # SLO failure (see analysis.end_metrics), because dropping work to make the
    # admitted set look fast would otherwise be trivially "optimal".
    shed: bool = False


def _build_engine(cfg: dict):
    """Construct AsyncLLMEngine with the patched scheduler. GPU-only path."""
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM  # v1 async engine

    m = cfg["model"]
    spec = None
    if m.get("spec_method") == "ngram":
        # prompt-lookup drafting: no separate draft model, so this must be
        # checked before the draft_model/eagle_model branches below (which
        # both require a model path and would otherwise leave spec=None).
        spec = {"method": "ngram",
                "num_speculative_tokens": cfg["runtime"].get("gamma_init", 4),
                "prompt_lookup_max": m.get("prompt_lookup_max", 4),
                "prompt_lookup_min": m.get("prompt_lookup_min", 2)}
    elif m.get("draft_model"):
        spec = {"method": m.get("spec_method", "draft_model"),
                "model": m["draft_model"],
                "num_speculative_tokens": cfg["runtime"].get("gamma_init", 4)}
    elif m.get("eagle_model"):
        spec = {"method": "eagle3", "model": m["eagle_model"],
                "num_speculative_tokens": cfg["runtime"].get("gamma_init", 4)}

    args = AsyncEngineArgs(
        model=m["target_model"],
        tensor_parallel_size=m.get("tp", 1),
        gpu_memory_utilization=m.get("gpu_mem_util", 0.90),
        max_model_len=m.get("max_model_len", 4096),
        max_num_seqs=cfg["runtime"].get("max_num_seqs_init", 64),
        dtype=m.get("dtype", "auto"),
        enforce_eager=m.get("enforce_eager", False),
        speculative_config=spec,
        # multi-turn replay (specloop_rt.multiturn) resubmits the whole
        # conversation-so-far each turn; prefix caching is what makes that
        # cheap instead of recomputing shared history from scratch. Off by
        # default so single-shot runs are unaffected unless a config opts in.
        enable_prefix_caching=m.get("enable_prefix_caching", False),
        # the patched scheduler:
        scheduler_cls="specloop_rt.vllm_patch.SpecLoopScheduler",
        disable_log_stats=False,
    )
    return AsyncLLM.from_engine_args(args)


async def _run_one(engine, req: W.TraceRequest, sampling, t0: float,
                   results: Dict[str, RequestResult],
                   shed_fn=None):
    from vllm import SamplingParams  # noqa
    # pace to arrival
    dt = req.arrival_s - (time.monotonic() - t0)
    if dt > 0:
        await asyncio.sleep(dt)
    submit = time.monotonic()
    rr = RequestResult(rid=req.rid, rtype=req.rtype, arrival_s=req.arrival_s,
                       submit_wall=submit)
    results[req.rid] = rr
    # Admission-time shedding: reject before the engine sees the request, so a
    # request that could not have met its TTFT SLO anyway does not also consume
    # KV blocks and slow down the requests that still can.
    if shed_fn is not None and shed_fn():
        rr.shed = True
        return
    sp = sampling(req.max_tokens)
    n = 0
    async for out in engine.generate(req.prompt, sp, request_id=req.rid):
        now = time.monotonic()
        toks = len(out.outputs[0].token_ids) if out.outputs else 0
        if rr.first_token_wall is None and toks > 0:
            rr.first_token_wall = now
            rr.ttft_s = now - submit
        n = toks
        if out.finished:
            rr.finish_wall = now
            rr.output_tokens = n
            rr.e2e_s = now - submit
            if n > 0 and rr.first_token_wall is not None:
                gen = now - rr.first_token_wall
                rr.tpot_s = gen / max(1, n - 1)
            break


async def replay(cfg: dict, trace: List[W.TraceRequest], out_dir: str):
    from vllm import SamplingParams
    os.makedirs(out_dir, exist_ok=True)

    controller = build_controller(cfg["controller"], cfg["runtime"]["tpot_slo_s"])
    telemetry = TelemetryWriter(os.path.join(out_dir, "steps.jsonl"),
                                run_meta={"cfg": cfg, "n_req": len(trace)})

    # install controller+telemetry BEFORE engine build (scheduler reads globals)
    from specloop_rt.vllm_patch import configure
    configure(controller, telemetry, cfg["runtime"])

    engine = _build_engine(cfg)

    def sampling(mt):
        return SamplingParams(temperature=cfg["runtime"].get("temperature", 0.0),
                              max_tokens=mt, ignore_eos=cfg["runtime"].get("ignore_eos", False))

    # Optional admission-time load shedding. Off unless runtime.shed_enabled.
    # Rejects an arrival whose expected TTFT already exceeds the SLO.
    #
    # The signal is deliberately CLIENT-side. An earlier version read the
    # scheduler's StepObservation via vllm_patch.last_observation(), which
    # silently never shed anything: vLLM v1 runs EngineCore in a separate
    # process (VLLM_ENABLE_V1_MULTIPROCESSING defaults on), so the scheduler
    # set that global in the engine process while this predicate ran in the
    # client process and read None on every call. The failure was invisible in
    # the metrics -- shed_frac was simply 0.0, indistinguishable from "the
    # policy chose not to shed" -- which is exactly why shed_frac is reported.
    #
    # What the client does own: how many requests it has in flight and what
    # TTFT the recently-admitted ones actually got. A rising observed TTFT is
    # the direct measurement of the quantity the SLO is written against, so no
    # cross-process plumbing is needed.
    shed_fn = None
    rt = cfg["runtime"]
    if rt.get("shed_enabled", False):
        shed_slo_mult = rt.get("shed_slo_mult", 1.0)
        ttft_slo = rt["ttft_slo_s"]
        budget = shed_slo_mult * ttft_slo

        def shed_fn() -> bool:  # noqa: F811
            now = time.monotonic()
            # TTFT of requests that have already produced a first token, most
            # recent first; EMA-free, just the last few, to stay responsive.
            recent = [r.ttft_s for r in results.values() if r.ttft_s is not None]
            if len(recent) >= 5:
                tail = recent[-20:]
                if sum(tail) / len(tail) > budget:
                    return True
            # Also shed on in-flight requests that are ALREADY past budget
            # without a first token -- catches a queue collapsing right now,
            # before any of its victims have reported a TTFT.
            stuck = sum(1 for r in results.values()
                        if r.first_token_wall is None and not r.shed
                        and r.finish_wall is None
                        and (now - r.submit_wall) > budget)
            return stuck >= 5

    results: Dict[str, RequestResult] = {}
    t0 = time.monotonic()
    tasks = [asyncio.create_task(_run_one(engine, r, sampling, t0, results, shed_fn))
             for r in trace]
    await asyncio.gather(*tasks)
    telemetry.close()

    with open(os.path.join(out_dir, "requests.jsonl"), "w") as f:
        for rr in results.values():
            f.write(json.dumps(asdict(rr)) + "\n")
    return results


TRACE_BUILDERS = {
    "step": W.step_perturbation, "mixed": W.mixed,
    "volatile": W.volatile, "homogeneous": W.homogeneous,
    "bursty_step": W.bursty_step_perturbation, "bursty_mixed": W.bursty_mixed,
    "bursty_volatile": W.bursty_volatile, "bursty_homogeneous": W.bursty_homogeneous,
}


def main(argv=None):
    p = argparse.ArgumentParser("specloop-replay")
    p.add_argument("--config", required=True)
    p.add_argument("--trace", default="mixed")
    p.add_argument("--rate", type=float, default=8.0)
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results_gpu/run")
    p.add_argument("--real-corpus", action="store_true",
                   help="draw prompts from ShareGPT/HumanEval/SQuAD/CNN-DailyMail "
                        "instead of synthetic templates (see specloop_rt.real_corpus)")
    p.add_argument("--force-output-tokens", type=int, default=None,
                   help="override the per-rtype lognormal output length with a fixed "
                        "value -- for the bound-edness sweep (rate x length -> "
                        "admission-bound vs decode-bound); pair with ignore_eos: true "
                        "in the config so generations run the full length")
    p.add_argument("--rtype", default=None,
                   help="restrict to a single rtype (rag/code/chat/reason) -- only "
                        "meaningful with --trace homogeneous/bursty_homogeneous; "
                        "for the axis-2 workload-type sweep")
    a = p.parse_args(argv)
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    builder = TRACE_BUILDERS[a.trace]
    kw = dict(rate=a.rate, seed=a.seed, use_real_corpus=a.real_corpus,
              force_output_tokens=a.force_output_tokens)
    if a.rtype is not None:
        if "homogeneous" not in a.trace:
            raise SystemExit("--rtype requires --trace homogeneous or bursty_homogeneous")
        kw["rtype"] = a.rtype
    if a.trace in ("step", "bursty_step"):
        kw.update(warmup=a.duration / 3, post=2 * a.duration / 3)
    else:
        kw.update(duration=a.duration)
    trace = builder(**kw)
    W.save_trace(trace, a.out + "_trace.jsonl") if not os.path.isdir(a.out) else None
    asyncio.run(replay(cfg, trace, a.out))
    print(f"done: {len(trace)} requests -> {a.out}")


if __name__ == "__main__":
    main()
