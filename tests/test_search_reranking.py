import numpy as np
import pytest

from app import search, embeddings, db, search_telemetry as telemetry
from app.schemas import Requirement, CandidateAssessment, SearchRequest
from app.provider import ProcessingError
from tests.test_api import client, fake_embed


def judgment(candidate_id, statuses, quotes=None):
    return CandidateAssessment(candidate_id=candidate_id, requirements=[
        {"requirement_index": i, "status": status, "evidence": (quotes or {}).get(i, [])}
        for i, status in enumerate(statuses)])


def test_soft_minimum_rewards_distributed_coverage():
    requirements = [Requirement(text="a"), Requirement(text="b"), Requirement(text="c")]
    scores = search.aggregate_requirements(np.array([[.95,.35,.35],[.54,.54,.54]]), requirements)
    assert scores[1] > scores[0]
    np.testing.assert_allclose(search.aggregate_requirements(np.array([[.3],[.8]]), requirements[:1]), [.3,.8])


def test_rank_contradictions_below_unknown_and_supported():
    requirements = [Requirement(text="Pilot"),Requirement(text="Overnight")]
    supported = judgment(1,["supported","supported"])
    unknown = judgment(2,["supported","unknown"])
    contradicted = judgment(3,["supported","contradicted"])
    assert search.ranking_key(supported,requirements,.4) < search.ranking_key(unknown,requirements,.8)
    assert search.ranking_key(unknown,requirements,.4) < search.ranking_key(contradicted,requirements,.99)


@pytest.mark.parametrize("fault", ["candidate", "duplicate", "requirement", "quote", "source", "missing_quote"])
def test_assessment_validation(fault):
    requirements=[Requirement(text="Pilot")]
    bundle=[{"candidate_id":1,"documents":[{"source_id":"resume","text":"Pilot in Dallas"}]}]
    result=search.Assessments(candidates=[judgment(1,["supported"],{0:[{"source_id":"resume","quote":"Pilot"}]})])
    item=result.candidates[0]
    if fault == "candidate": item.candidate_id=2
    elif fault == "duplicate": result.candidates.append(item)
    elif fault == "requirement": item.requirements[0].requirement_index=1
    elif fault == "quote": item.requirements[0].evidence[0].quote="invented"
    elif fault == "source": item.requirements[0].evidence[0].source_id="note:999"
    else: item.requirements[0].evidence=[]
    with pytest.raises(ProcessingError): search.validate_assessments(result,requirements,bundle)


def test_evidence_caps_preserve_notes_and_disclose_truncation():
    bundle=search.evidence_bundle({"id":1,"resume":"x"*20000},[
        {"id":2,"note":"Cannot relocate", "timestamp":"2026-01-01T00:00:00Z"}])
    assert bundle["evidence_truncated"]
    assert bundle["documents"][1]["text"] == "Cannot relocate"
    assert sum(len(d["text"]) for d in bundle["documents"]) <= search.EVIDENCE_CHARS


def test_api_uses_cross_document_evidence_and_returns_assessments(client, monkeypatch):
    client.post('/candidates/1/notes',json={"note":"Available overnight"})
    monkeypatch.setattr(search,"plan_query",lambda _:search.QueryPlan(requirements=[Requirement(text="Captain"),Requirement(text="Available overnight")]))
    monkeypatch.setattr(embeddings,"embed_queries",fake_embed)
    def assess(query, requirements, bundles):
        assert len(bundles)<=search.SHORTLIST_SIZE
        pilot=next(b for b in bundles if b["candidate_id"]==1)
        assert any(d["text"]=="Available overnight" for d in pilot["documents"])
        return search.Assessments(candidates=[judgment(b["candidate_id"], ["supported","supported"] if b["candidate_id"]==1 else ["unknown","unknown"]) for b in bundles])
    monkeypatch.setattr(search,"assess",assess)
    response=client.post('/search',json={"strategy":"reranked","query":"Captain available overnight","top_k":2})
    assert response.status_code==200
    result=response.json()
    assert result["results"][0]["id"]==1 and result["strategy"]=="reranked"
    assert len(result["requirements"])==2 and result["shortlist_size"]==3
    assert result["results"][0]["relevance_grade"] == 4
    assert result["results"][1]["relevance_grade"] == 1


def test_empty_filters_skip_all_provider_calls(client, monkeypatch):
    monkeypatch.setattr(search,"plan_query",lambda _:pytest.fail('unnecessary LLM call'))
    result=client.post('/search',json={"strategy":"reranked","query":"pilot","location_city":"no such city"})
    assert result.status_code==200 and result.json()["results"]==[]


def test_failure_does_not_silently_fall_back(client, monkeypatch):
    def fail(_): raise ProcessingError('bad output')
    monkeypatch.setattr(search,"plan_query",fail)
    assert client.post('/search',json={"strategy":"reranked","query":"pilot"}).status_code==503


def test_one_validation_retry_is_bounded(monkeypatch):
    calls=[]
    def invalid(*args):
        calls.append(args[-1]); return search.Assessments(candidates=[])
    monkeypatch.setattr(search,"run_agent",invalid)
    with pytest.raises(ProcessingError):
        search.assess('pilot',[Requirement(text='pilot')],[{"candidate_id":1,"documents":[]}])
    assert calls==["reranking","reranking_retry"]


def test_cost_accounting_and_context_isolation():
    with telemetry.capture() as first:
        telemetry.record_call('gemini-3.5-flash-lite','reranking',1000,200)
    assert first['estimated_cost_usd']==pytest.approx(.0008)
    with telemetry.capture() as second: pass
    assert second['calls']==[] and second['estimated_cost_usd']==0


def test_defaults_to_new_pipeline():
    assert SearchRequest(query='pilot').strategy=='reranked'


def test_fixed_shortlist_even_when_more_results_requested(client, monkeypatch):
    row = db.get_candidate(1)
    extra = [{**row, "name": f"Extra {i}", "resume": f"Pilot resume {i}"} for i in range(15)]
    db.insert_candidates(extra, fake_embed([r['resume'] for r in extra]))
    monkeypatch.setattr(search, 'plan_query', lambda _: search.QueryPlan(requirements=[Requirement(text='pilot')]))
    monkeypatch.setattr(embeddings, 'embed_queries', fake_embed)
    def assess(query, requirements, bundles):
        assert len(bundles) == 12
        return search.Assessments(candidates=[judgment(b['candidate_id'], ['unknown']) for b in bundles])
    monkeypatch.setattr(search, 'assess', assess)
    result = client.post('/search', json={'strategy':'reranked','query':'pilot','top_k':100}).json()
    assert len(result['results']) == 12 and result['shortlist_size'] == 12
    assert result['warning'] and result['total_matching_filters'] == 18


def test_failed_eval_marks_cost_incomplete():
    from app.evaluation import EvalDataset, run_dataset
    dataset = EvalDataset.model_validate({'name':'failure','description':'test','cases':[
        {'id':'one','kind':'retrieval','search':{'query':'pilot'},'relevant_ids':[1],'rationale':'test'}]})
    def fail(_): raise ProcessingError('failure')
    report = run_dataset(dataset, fail, lambda _: None)
    assert not report['summary']['cost_complete']
    assert report['summary']['errors'] == 1


@pytest.mark.parametrize("mandatory,statuses,expected", [
    ([True, True], ["supported", "supported"], 4),
    ([True, False], ["supported", "unknown"], 3),
    ([True, False], ["supported", "contradicted"], 3),
    ([True, True], ["supported", "unknown"], 2),
    ([True, True], ["unknown", "unknown"], 1),
    ([True, True], ["supported", "contradicted"], 0),
    ([True, False], ["unknown", "supported"], 1),
    ([False, False], ["supported", "supported"], 4),
    ([False, False], ["supported", "unknown"], 3),
    ([False, False], ["supported", "contradicted"], 2),
    ([False, False], ["contradicted", "unknown"], 1),
    ([True], ["supported"], 4),
    ([True], ["unknown"], 1),
    ([True], ["contradicted"], 0),
])
def test_relevance_rubric(mandatory, statuses, expected):
    requirements = [Requirement(text=f"Requirement {i}", mandatory=value) for i, value in enumerate(mandatory)]
    assert search.relevance_grade(judgment(1, statuses), requirements) == expected


def test_grade_beats_retrieval_similarity():
    requirements = [Requirement(text="pilot"), Requirement(text="overnight")]
    candidates = [judgment(1,["supported","contradicted"]), judgment(2,["unknown","unknown"]), judgment(3,["supported","supported"])]
    assert [c.candidate_id for c in sorted(candidates, key=lambda c: search.ranking_key(c, requirements, 1/c.candidate_id))] == [3,2,1]


def test_candidate_lookup_is_literal_and_opens_existing_notes(client):
    assert [c["id"] for c in client.get('/candidates', params={"name":"  pILot  "}).json()] == [1]
    assert len(client.get('/candidates',params={"name":"Person"}).json()) == 3
    for name in ["missing", "%", "_", "' OR 1=1 --"]:
        assert client.get('/candidates',params={"name":name}).json() == []
    assert len(client.get('/candidates',params={"name":" "}).json()) == 3
    assert [c["name"] for c in client.get("/candidates").json()] == ["Agent Person", "Mechanic Person", "Pilot Person"]
    assert client.get('/candidates',params={"name":"x"*201}).status_code == 422
    assert client.post('/candidates/1/notes',json={"note":"Ready to start"}).status_code == 201
    assert client.get('/candidates/1').json()["notes"][0]["note"] == "Ready to start"


def test_provider_token_breakdown_does_not_double_charge():
    with telemetry.capture() as trace:
        telemetry.record_call('gemini-3.5-flash-lite','reranking',1000,200,
                              usage_details={"thoughts_tokens":50,"answer_tokens":150})
        telemetry.record_call('gemini-3.5-flash-lite','requirements',10,30,
                              usage_details={"answer_tokens":30})
        telemetry.record_call('unknown','missing',10,30)
    first, second, third = trace['calls']
    assert (first['thinking_tokens'],first['answer_tokens']) == (50,150)
    assert first['estimated_cost_usd'] == pytest.approx(.0008)
    assert second['thinking_tokens'] == 0 and second['answer_tokens'] == 30
    assert third['thinking_tokens'] is None and third['answer_tokens'] is None


def test_google_adapter_preserves_raw_usage(monkeypatch):
    from types import SimpleNamespace
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.usage import RequestUsage
    monkeypatch.setattr(search.GoogleModel,"_process_response",lambda *args: ModelResponse(parts=[],usage=RequestUsage(output_tokens=23)))
    model=object.__new__(search.UsageGoogleModel)
    response=SimpleNamespace(usage_metadata=SimpleNamespace(candidates_token_count=20,thoughts_token_count=3))
    result=model._process_response(response,None)
    assert result.usage.details == {"answer_tokens":20,"thoughts_tokens":3}
