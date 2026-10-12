"""Request-local timings and token accounting; never captures source text."""
from contextvars import ContextVar
from contextlib import contextmanager
from time import perf_counter

PRICING = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-embedding-2": {"input": 0.20, "output": 0.0},
}
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_DATE = "2026-10-12"
_current = ContextVar("search_trace", default=None)


@contextmanager
def capture():
    trace = {"stages_ms": {}, "calls": [], "shortlist_ids": []}
    token = _current.set(trace)
    start = perf_counter()
    try:
        yield trace
    finally:
        trace["total_ms"] = (perf_counter()-start)*1000
        trace["estimated_cost_usd"] = sum(c["estimated_cost_usd"] or 0 for c in trace["calls"])
        trace["cost_complete"] = all(c["estimated_cost_usd"] is not None for c in trace["calls"]) and not trace.get("failed", False)
        trace["pricing"] = {"source": PRICING_SOURCE, "checked": PRICING_DATE, "usd_per_million": PRICING}
        _current.reset(token)


@contextmanager
def stage(name):
    start = perf_counter()
    try:
        yield
    finally:
        trace = _current.get()
        if trace is not None:
            trace["stages_ms"][name] = trace["stages_ms"].get(name, 0)+(perf_counter()-start)*1000


def record_call(model, stage_name, input_tokens, output_tokens=0, basis="provider", requests=1, usage_details=None):
    trace = _current.get()
    if trace is None:
        return
    details = usage_details or {}
    thinking = details.get("thoughts_tokens")
    answer = output_tokens - thinking if thinking is not None else details.get("answer_tokens", details.get("text_candidates_tokens"))
    if thinking is None and answer is not None:
        thinking = output_tokens - answer
    price = PRICING.get(model)
    cost = (input_tokens*price["input"]+output_tokens*price["output"])/1_000_000 if price else None
    trace["calls"].append(dict(model=model, stage=stage_name, input_tokens=input_tokens,
        output_tokens=output_tokens, thinking_tokens=thinking, answer_tokens=answer,
        token_breakdown_basis="provider" if thinking is not None else "unavailable",
        token_basis=basis, requests=requests, estimated_cost_usd=cost))


def shortlist(ids):
    trace = _current.get()
    if trace is not None:
        trace["shortlist_ids"] = ids
