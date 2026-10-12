"""Run draft/reviewed eval cases against the current DB and configured live Gemini models."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, UTC
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import db, preextract
from app.config import GEMINI_MODEL, EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_FORMAT_VERSION
from app.evaluation import EvalDataset, run_dataset, summarise
from app.main import search
from app.search import SHORTLIST_SIZE, EVIDENCE_CHARS, PIPELINE_VERSION, PLAN_PROMPT, RERANK_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/starter.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/results/latest.json"))
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--retry-errors", type=Path, help="rerun only failed cases; preserve attempts and cumulative cost/time")
    parser.add_argument("--strategy", choices=["baseline", "reranked"], default="reranked")
    parser.add_argument("--validate-only", action="store_true", help="validate cases and candidate IDs without calling Gemini")
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    dataset = EvalDataset.model_validate_json(raw)
    for case in dataset.cases:
        if case.search is not None:
            case.search.strategy = args.strategy
    available = db.get_candidate_ids()
    missing = {i for case in dataset.cases for i in case.relevant_ids + case.contradicted_ids} - available
    if missing:
        parser.error(f"Judgments reference missing candidate IDs: {sorted(missing)}")
    if args.validate_only:
        print(f"Valid: {len(dataset.cases)} cases ({dataset.judgments} judgments)")
        return 0
    db.assert_embedding_config()
    before = db.state_token()
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    if baseline:
        for key, value in {"dataset_sha256":hashlib.sha256(raw).hexdigest(), "database_fingerprint":before}.items():
            if baseline["metadata"][key] != value:
                parser.error(f"Cannot compare runs with different {key}")
    previous = json.loads(args.retry_errors.read_text()) if args.retry_errors else None
    if previous:
        for key, value in {"dataset_sha256":hashlib.sha256(raw).hexdigest(), "database_fingerprint":before, "ranking":args.strategy, "extraction_model":GEMINI_MODEL, "embedding_model":EMBEDDING_MODEL,
                           "prompt_sha256":hashlib.sha256((PLAN_PROMPT+RERANK_PROMPT).encode()).hexdigest()}.items():
            if previous["metadata"][key] != value:
                parser.error(f"Cannot retry a run with different {key}")
        failed_ids = {r["id"] for r in previous["cases"] if r["error"]}
        if not failed_ids:
            parser.error("The selected report has no failed cases")
        subset = dataset.model_copy(update={"cases":[c for c in dataset.cases if c.id in failed_ids]})
        retried = run_dataset(subset, search, preextract.extract_draft)
        by_id = {r["id"]:r for r in retried["cases"]}
        merged=[]
        for old in previous["cases"]:
            if old["id"] not in by_id:
                merged.append(old)
                continue
            new = by_id[old["id"]]
            new["previous_attempts"] = old.get("previous_attempts", []) + [{k:v for k,v in old.items() if k != "previous_attempts"}]
            new["latency_ms"] += old["latency_ms"]
            new["telemetry"]["estimated_cost_usd"] += old["telemetry"]["estimated_cost_usd"]
            new["telemetry"]["cost_complete"] = False  # Failed requests may have unreported charges.
            merged.append(new)
        report = summarise(dataset, merged)
        report["summary"]["prior_failed_attempts"] = sum(len(r.get("previous_attempts", [])) for r in merged)
    else:
        report = run_dataset(dataset, search, preextract.extract_draft)
    report["metadata"] = {"timestamp": datetime.now(UTC).isoformat(), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "database_fingerprint": before, "database_unchanged": before == db.state_token(),
        "extraction_model": GEMINI_MODEL, "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIM, "embedding_format": EMBEDDING_FORMAT_VERSION,
        "ranking": args.strategy, "pipeline_version": PIPELINE_VERSION,
        "shortlist_size": SHORTLIST_SIZE, "evidence_chars_per_candidate": EVIDENCE_CHARS,
        "prompt_sha256": hashlib.sha256((PLAN_PROMPT+RERANK_PROMPT).encode()).hexdigest()}
    if args.compare:
        for key in ("dataset_sha256", "database_fingerprint"):
            if baseline["metadata"][key] != report["metadata"][key]:
                parser.error(f"Cannot compare runs with different {key}")
        report["delta"] = {k: v-baseline["summary"][k] for k,v in report["summary"].items() if k in baseline["summary"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {args.output}; judgments: {dataset.judgments}")
    return int(report["summary"]["errors"] > 0 or not report["metadata"]["database_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
