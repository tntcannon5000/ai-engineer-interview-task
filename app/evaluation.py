"""Small deterministic metrics around live retrieval and extraction, no LLM judge."""
from math import log2
from time import perf_counter
from typing import Literal
from pydantic import Field, model_validator
from app.schemas import StrictInput, SearchRequest
from app import search_telemetry as telemetry


class EvalCase(StrictInput):
    id: str
    kind: Literal["retrieval", "extraction"]
    search: SearchRequest | None = None
    text: str | None = None
    relevant_ids: list[int] = Field(default_factory=list)
    contradicted_ids: list[int] = Field(default_factory=list)
    expected_fields: dict = Field(default_factory=dict)
    rationale: str

    @model_validator(mode="after")
    def valid_case(self):
        from app.preextract import CandidateDraft
        if self.kind == "retrieval":
            if self.search is None or self.text is not None or self.expected_fields:
                raise ValueError("Retrieval cases require search and relevant_ids only")
            if len(set(self.relevant_ids)) != len(self.relevant_ids) or any(i <= 0 for i in self.relevant_ids):
                raise ValueError("Relevant IDs must be positive and unique")
            if len(set(self.contradicted_ids)) != len(self.contradicted_ids) or any(i <= 0 for i in self.contradicted_ids):
                raise ValueError("Contradicted IDs must be positive and unique")
            if set(self.contradicted_ids) & set(self.relevant_ids):
                raise ValueError("Positive and contradicted judgments must not overlap")
        else:
            if not self.text or not self.expected_fields or self.search is not None or self.relevant_ids or self.contradicted_ids:
                raise ValueError("Extraction cases require text and expected_fields only")
            if set(self.expected_fields) - CandidateDraft.model_fields.keys():
                raise ValueError("Unknown expected extraction fields")
        return self


class EvalDataset(StrictInput):
    name: str
    judgments: Literal["draft", "reviewed"] = "draft"
    description: str
    cases: list[EvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Case IDs must be unique")
        return self


def retrieval_metrics(actual, relevant, k):
    ranked = actual[:k]
    relevant = set(relevant)
    hits = [int(i in relevant) for i in ranked]
    if not relevant:
        return {"empty_result_correct": float(not ranked)}
    ideal = sum(1/log2(i+2) for i in range(min(k, len(relevant))))
    return {
        "recall_at_k": sum(hits)/len(relevant),
        "precision_at_k": sum(hits)/k,
        "mrr": next((1/(i+1) for i, hit in enumerate(hits) if hit), 0.0),
        "ndcg_at_k": sum(hit/log2(i+2) for i, hit in enumerate(hits))/ideal,
    }


def run_dataset(dataset, search_fn, extract_fn):
    results = []
    for case in dataset.cases:
        start = perf_counter()
        with telemetry.capture() as trace:
            result = run_case(case, search_fn, extract_fn, start)
            trace["failed"] = result["error"] is not None
        result["telemetry"] = trace
        if case.kind == "retrieval" and case.relevant_ids:
            result["metrics"]["shortlist_recall"] = len(set(trace["shortlist_ids"]) & set(case.relevant_ids))/len(case.relevant_ids)
        results.append(result)
    return summarise(dataset, results)


def run_case(case, search_fn, extract_fn, start):
    try:
        if case.kind == "retrieval":
            response = search_fn(case.search)
            actual = [r.id for r in response.results]
            metrics = retrieval_metrics(actual, case.relevant_ids, case.search.top_k) if case.relevant_ids or not case.contradicted_ids else {}
            if case.contradicted_ids:
                metrics["contradicted_in_top_k"] = len(set(actual) & set(case.contradicted_ids))/len(case.contradicted_ids)
        else:
            actual = extract_fn(case.text).model_dump()
            matches = {key: actual.get(key) == value for key, value in case.expected_fields.items()}
            metrics = {"field_accuracy": sum(matches.values())/len(matches), "all_fields_correct": float(all(matches.values()))}
        return {"id": case.id, "kind": case.kind, "actual": actual, "metrics": metrics,
                "response": response.model_dump(mode="json") if case.kind == "retrieval" else None,
                "latency_ms": (perf_counter()-start)*1000, "error": None}
    except Exception as exc:
        chain = []
        cause = exc
        while cause is not None and len(chain) < 5:
            chain.append({"type": type(cause).__name__, "status_code": getattr(cause, "status_code", None)})
            cause = cause.__cause__
        # Retain a failed case without leaking provider messages or candidate text.
        return {"id": case.id, "kind": case.kind, "metrics": {},
                "latency_ms": (perf_counter()-start)*1000, "error": type(exc).__name__, "error_details": chain}

def summarise(dataset, results):
    keys = {key for row in results for key in row["metrics"]}
    summary = {key: sum(r["metrics"][key] for r in results if key in r["metrics"])/sum(key in r["metrics"] for r in results) for key in sorted(keys)}
    summary.update(cases=len(results), errors=sum(r["error"] is not None for r in results),
                   mean_latency_ms=sum(r["latency_ms"] for r in results)/len(results))
    latencies = sorted(r["latency_ms"] for r in results)
    summary.update(total_latency_ms=sum(latencies), p50_latency_ms=latencies[len(latencies)//2],
        p95_latency_ms=latencies[min(len(latencies)-1, int(len(latencies)*.95))],
        estimated_cost_usd=sum(r["telemetry"]["estimated_cost_usd"] for r in results),
        cost_complete=all(r["telemetry"]["cost_complete"] for r in results))
    return {"dataset": dataset.name, "judgments": dataset.judgments, "summary": summary, "cases": results,
            "cost": None, "cost_note": "Actual billing is not measured. estimated_cost_usd uses published paid-tier rates and recorded/labelled estimated tokens; failed requests may have unreported charges. Metrics average successful applicable cases; inspect errors separately."}
