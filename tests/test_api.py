import json
import hashlib

import pytest
from fastapi.testclient import TestClient

import re

import numpy as np

from app import db, embeddings, extraction
from app.config import EMBEDDING_DIM
from app.main import app
from app.schemas import CandidateFields
from app.ingest import build_database

FIXTURE = [
    {
        "name": "Pilot Person",
        "resume": "Captain flying Boeing 737 for a major US airline. ATP certificate, crew resource management.",
        "skills": ["ATP Certificate", "Type Rating B737"],
        "current_job_title": "Captain",
        "past_job_titles": ["First Officer"],
        "years_experience": 12,
        "location_city": "Dallas",
        "location_region": "Texas",
    },
    {
        "name": "Mechanic Person",
        "resume": "Aircraft maintenance technician repairing hydraulic systems and turbine engines. A&P licensed.",
        "skills": ["A&P License", "Hydraulic Systems"],
        "current_job_title": "Aircraft Maintenance Technician",
        "past_job_titles": ["Apprentice Mechanic"],
        "years_experience": 5,
        "location_city": "Denver",
        "location_region": "Colorado",
    },
    {
        "name": "Agent Person",
        "resume": "Customer service agent handling passenger check-in and rebooking with Sabre.",
        "skills": ["Sabre", "Passenger Check-In"],
        "current_job_title": "Customer Service Agent",
        "past_job_titles": ["Retail Associate"],
        "years_experience": 2,
        "location_city": "Dallas",
        "location_region": "Texas",
    },
]


def fake_embed(texts: list[str]) -> np.ndarray:
    """Hashed bag-of-words vectors: shared words => higher cosine. Keeps tests offline."""
    out = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z&]+", t.lower()):
            out[i, int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest()) % EMBEDDING_DIM] += 1.0
        if not out[i].any():
            out[i, 0] = 1.0  # Placeholder text still receives a valid vector from the fake provider.
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "embed_documents", fake_embed)
    monkeypatch.setattr(embeddings, "embed_query", lambda q: fake_embed([q])[0])
    resumes = tmp_path / "resumes.json"
    resumes.write_text(json.dumps(FIXTURE))
    monkeypatch.setattr("app.ingest.RESUMES_PATH", resumes)
    monkeypatch.setattr(db, "_db_path", tmp_path / "test.db")
    # Existing regression cases exercise the unchanged semantic baseline.
    class BaselineClient(TestClient):
        def post(self, url, **kwargs):
            if url == "/search" and isinstance(kwargs.get("json"), dict):
                kwargs["json"] = {"strategy": "baseline", **kwargs["json"]}
            return super().post(url, **kwargs)
    with BaselineClient(app) as c:
        assert build_database()["inserted"] == 3
        yield c


def test_semantic_ranking(client):
    r = client.post("/search", json={"query": "someone who fixes aircraft engines", "top_k": 1}).json()
    assert r["total_matching_filters"] == 3
    assert r["results"][0]["name"] == "Mechanic Person"


def test_filters(client):
    r = client.post("/search", json={"query": "anything", "location_region": "texas"}).json()
    assert {x["name"] for x in r["results"]} == {"Pilot Person", "Agent Person"}

    r = client.post("/search", json={"query": "anything", "skills": ["Sabre"]}).json()
    assert [x["name"] for x in r["results"]] == ["Agent Person"]

    r = client.post("/search", json={"query": "anything", "past_job_titles": ["First Officer"], "current_job_title": "Captain"}).json()
    assert [x["name"] for x in r["results"]] == ["Pilot Person"]

    r = client.post("/search", json={"query": "anything", "location_city": "Nowhere"}).json()
    assert r["results"] == [] and r["total_matching_filters"] == 0

    assert client.post("/search", json={"query": ""}).status_code == 422


def test_text_filters_are_substring_matches(client):
    r = client.post("/search", json={"query": "x", "current_job_title": "captain"}).json()
    assert [x["name"] for x in r["results"]] == ["Pilot Person"]  # "Captain"

    r = client.post("/search", json={"query": "x", "current_job_title": "Technician"}).json()
    assert [x["name"] for x in r["results"]] == ["Mechanic Person"]  # "Aircraft Maintenance Technician"

    r = client.post("/search", json={"query": "x", "past_job_titles": ["officer"]}).json()
    assert [x["name"] for x in r["results"]] == ["Pilot Person"]  # "First Officer"

    r = client.post("/search", json={"query": "x", "location_region": "tex"}).json()
    assert {x["name"] for x in r["results"]} == {"Pilot Person", "Agent Person"}

    r = client.post("/search", json={"query": "x", "location_city": "100%"}).json()  # wildcard is escaped
    assert r["total_matching_filters"] == 0

    r = client.post("/search", json={"query": "anything", "min_years_experience": 5}).json()
    assert {x["name"] for x in r["results"]} == {"Pilot Person", "Mechanic Person"}
    r = client.post("/search", json={"query": "anything", "min_years_experience": 3, "max_years_experience": 10}).json()
    assert [x["name"] for x in r["results"]] == ["Mechanic Person"]


def test_add_candidate_uses_extraction_and_taxonomy(client, monkeypatch):
    def fake_extract(resume: str) -> CandidateFields:
        return CandidateFields(
            skills=["Tower Control", "Not A Real Skill"],
            current_job_title="Air Traffic Controller",
            past_job_titles=["Developmental Controller"],
            years_experience=7,
            location_city="Chicago",
            location_region="Illinois",
        )

    monkeypatch.setattr(extraction, "extract_fields", fake_extract)
    r = client.post("/candidates", json={"name": "ATC Person", "resume": "Works the tower at ORD."})
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == 4
    assert body["skills"] == ["Tower Control"]

    assert client.get("/candidates/4").json()["name"] == "ATC Person"
    r = client.post("/search", json={"query": "control tower", "current_job_title": "Air Traffic Controller"}).json()
    assert [x["name"] for x in r["results"]] == ["ATC Person"]


def test_normalize_helpers():
    from app.taxonomy import normalize_skills

    assert normalize_skills(["a&p license", "Sabre", "Sabre", "bogus"]) == ["A&P License", "Sabre"]
    assert extraction.normalize_state("tx") == "Texas"
    assert extraction.normalize_state("Texas") == "Texas"
    assert extraction.normalize_state("Ontario") == "Ontario"
