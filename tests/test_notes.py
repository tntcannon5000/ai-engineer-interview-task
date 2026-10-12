import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db, embeddings, extraction
from app.config import EMBEDDING_DIM
from app.ingest import import_notes, build_database
from app.main import app
from app.schemas import CandidateFields, ImportNote
from tests.test_api import FIXTURE, client, fake_embed


@pytest.fixture
def notes_client(client, monkeypatch):
    monkeypatch.setattr(
        extraction, "extract_fields", lambda resume: CandidateFields.model_validate(FIXTURE[0])
    )
    return client


def note_count():
    with db.connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM candidate_notes").fetchone()[0]


def note_row(candidate_id, text, timestamp="2026-01-01T00:00:00Z"):
    return ImportNote(candidate_id=candidate_id, note=text, timestamp=timestamp).model_dump(mode="json")


def vector(score):
    result = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    result[:2] = [score, np.sqrt(1 - score**2)]
    return result


def set_resume_scores(scores):
    with db.connect() as conn:
        for candidate_id, score in enumerate(scores, start=1):
            conn.execute("UPDATE candidates SET embedding = ? WHERE id = ?", (db.to_blob(vector(score)), candidate_id))


def test_create_candidate_with_separate_initial_notes(notes_client, monkeypatch):
    calls = []
    def embed(texts):
        calls.append(texts)
        return fake_embed(texts)
    monkeypatch.setattr(embeddings, "embed_documents", embed)
    payload = {
        "name": "New Pilot", "resume": "Captain flying a Boeing 737.",
        "notes": [
            {"note": " Available for seasonal work in Anchorage. ", "timestamp": "2026-01-01T05:00:00+05:00"},
            {"note": "Open to evening shifts."},
        ],
    }
    response = notes_client.post("/candidates", json=payload)
    assert response.status_code == 201
    candidate = response.json()
    assert candidate["id"] == 4
    assert calls == [[payload["resume"], *(note["note"] for note in payload["notes"])]]
    assert candidate["notes"][0]["timestamp"] == "2026-01-01T00:00:00Z"
    assert candidate["notes"][0]["note"] == payload["notes"][0]["note"]
    assert all(note["candidate_id"] == candidate["id"] for note in candidate["notes"])
    assert notes_client.get("/candidates/4").json()["notes"] == candidate["notes"]
    assert note_count() == 2


def test_add_note_preserves_resume_and_becomes_searchable(client, monkeypatch):
    with db.connect() as conn:
        original = bytes(conn.execute("SELECT embedding FROM candidates WHERE id = 1").fetchone()[0])
    monkeypatch.setattr(extraction, "extract_fields", lambda _: pytest.fail("Note addition must not extract resume fields"))
    text = "Available for seasonal work in Anchorage."
    response = client.post("/candidates/1/notes", json={"note": text})
    assert response.status_code == 201
    saved = response.json()
    assert saved["candidate_id"] == 1
    assert saved["timestamp"].endswith("Z")
    result = client.post("/search", json={"query": text, "top_k": 1}).json()
    assert result["results"][0]["id"] == 1
    assert result["results"][0]["match_source"] == "note"
    assert result["results"][0]["matched_note"] == saved
    with db.connect() as conn:
        assert conn.execute("SELECT embedding FROM candidates WHERE id = 1").fetchone()[0] == original
        assert len(conn.execute("SELECT embedding FROM candidate_notes").fetchone()[0]) == EMBEDDING_DIM * 4


@pytest.mark.parametrize("payload", [
    {}, {"note": "   "}, {"note": "text", "timestamp": "invalid"},
    {"note": "text", "timestamp": "2026-01-01T00:00:00"},
])
def test_invalid_notes_rejected_before_provider_calls(client, monkeypatch, payload):
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Invalid input must not call Gemini"))
    assert client.post("/candidates/1/notes", json=payload).status_code == 422
    assert client.post("/candidates", json={"name": "New", "resume": "Resume", "notes": [payload]}).status_code == 422
    assert note_count() == 0


def test_unknown_candidate_rejected_before_embedding(client, monkeypatch):
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Unknown candidate must not call Gemini"))
    assert client.post("/candidates/999/notes", json={"note": "Available evenings"}).status_code == 404


def test_duplicate_note_is_conflict(client):
    payload = {"note": "Available evenings", "timestamp": "2026-01-01T00:00:00Z"}
    assert client.post("/candidates/1/notes", json=payload).status_code == 201
    assert client.post("/candidates/1/notes", json=payload).status_code == 409
    assert note_count() == 1


def test_candidate_notes_order_handles_fractional_seconds(client):
    for timestamp in ["2026-01-01T00:00:00.100000Z", "2026-01-01T00:00:00Z"]:
        assert client.post("/candidates/1/notes", json={"note": "Availability", "timestamp": timestamp}).status_code == 201
    notes = client.get("/candidates/1").json()["notes"]
    assert [note["timestamp"] for note in notes] == ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00.100000Z"]


def test_provider_failure_leaves_candidate_and_notes_unchanged(notes_client, monkeypatch):
    def fail(_):
        raise RuntimeError("Embedding failure")
    monkeypatch.setattr(embeddings, "embed_documents", fail)
    assert notes_client.post("/candidates/1/notes", json={"note": "Available evenings"}).status_code == 503
    assert notes_client.post("/candidates", json={
        "name": "New", "resume": "Resume", "notes": [{"note": "Available evenings"}],
    }).status_code == 503
    assert db.count_candidates() == 3
    assert note_count() == 0


def test_initial_note_conflict_rolls_back_entire_candidate(notes_client):
    duplicate = {"note": "Available evenings", "timestamp": "2026-01-01T00:00:00Z"}
    response = notes_client.post("/candidates", json={
        "name": "New", "resume": "Resume", "notes": [duplicate, duplicate],
    })
    assert response.status_code == 409
    assert db.count_candidates() == 3
    assert note_count() == 0


def test_search_max_score_deduplicates_candidates_and_keeps_filters(client, monkeypatch):
    set_resume_scores([0.8, 0.6, 0.4])
    rows = [note_row(1, f"Note {i}") for i in range(3)] + [note_row(2, "Best note")]
    notes = db.insert_notes(rows, np.stack([vector(0.85)] * 3 + [vector(0.9)]))
    monkeypatch.setattr(embeddings, "embed_query", lambda _: vector(1))
    response = client.post("/search", json={"query": "query"}).json()
    assert response["total_matching_filters"] == 3
    assert [result["id"] for result in response["results"]] == [2, 1, 3]
    assert [result["score"] for result in response["results"]] == pytest.approx([0.9, 0.85, 0.4])
    assert response["results"][0]["matched_note"]["id"] == notes[-1]["id"]
    assert response["results"][2]["match_source"] == "resume"
    assert response["results"][2]["matched_note"] is None
    filtered = client.post("/search", json={"query": "query", "location_region": "Texas"}).json()
    assert filtered["total_matching_filters"] == 2
    assert [result["id"] for result in filtered["results"]] == [1, 3]
    assert len(client.post("/search", json={"query": "query", "top_k": 1}).json()["results"]) == 1


def test_search_ties_prefer_resume_and_stable_candidate_order(client, monkeypatch):
    set_resume_scores([0.7, 0.7, 0.7])
    db.insert_notes([note_row(1, "Equal note")], np.stack([vector(0.7)]))
    monkeypatch.setattr(embeddings, "embed_query", lambda _: vector(1))
    results = client.post("/search", json={"query": "query"}).json()["results"]
    assert [result["id"] for result in results] == [1, 2, 3]
    assert all(result["match_source"] == "resume" for result in results)


def test_search_negative_scores_are_not_clamped(client, monkeypatch):
    set_resume_scores([-0.2, -0.3, -0.4])
    db.insert_notes([note_row(2, "Less negative note")], np.stack([vector(-0.1)]))
    monkeypatch.setattr(embeddings, "embed_query", lambda _: vector(1))
    results = client.post("/search", json={"query": "query"}).json()["results"]
    assert [result["id"] for result in results] == [2, 1, 3]
    assert results[0]["score"] == pytest.approx(-0.1)


def test_import_skips_existing_and_repeated_notes_without_reembedding(client, tmp_path, monkeypatch):
    existing = note_row(1, "Existing note")
    assert client.post("/candidates/1/notes", json={key: value for key, value in existing.items() if key != "candidate_id"}).status_code == 201
    new = note_row(2, "New note")
    equivalent_timestamp = {**new, "timestamp": "2025-12-31T19:00:00-05:00"}
    path = tmp_path / "notes.json"
    path.write_text(json.dumps([existing, new, equivalent_timestamp, note_row(3, "Another note")]), encoding="utf-8")
    calls = []
    def embed(texts):
        calls.append(texts)
        return fake_embed(texts)
    monkeypatch.setattr(embeddings, "embed_documents", embed)
    assert import_notes(path) == {"inserted": 2, "skipped": 2}
    assert calls == [["New note", "Another note"]]
    assert import_notes(path) == {"inserted": 0, "skipped": 4}
    assert len(calls) == 1
    assert note_count() == 3


def test_import_rejects_unknown_candidates_before_embedding(client, tmp_path, monkeypatch):
    path = tmp_path / "notes.json"
    path.write_text(json.dumps([note_row(1, "Valid candidate"), note_row(999, "Unknown candidate")]))
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Invalid import must not call Gemini"))
    with pytest.raises(ValueError, match="Unknown candidate IDs"):
        import_notes(path)
    assert note_count() == 0


def test_import_provider_failure_preserves_existing_data(client, tmp_path, monkeypatch):
    assert client.post("/candidates/1/notes", json={"note": "Existing note"}).status_code == 201
    path = tmp_path / "notes.json"
    path.write_text(json.dumps([note_row(2, "New note"), note_row(3, "Another note")]))
    def fail(_):
        raise RuntimeError("Embedding failure")
    monkeypatch.setattr(embeddings, "embed_documents", fail)
    with pytest.raises(RuntimeError):
        import_notes(path)
    assert note_count() == 1
    assert db.count_candidates() == 3


def test_startup_adds_notes_table_without_rebuilding_resumes(client, monkeypatch):
    with db.connect() as conn:
        before = [tuple(row) for row in conn.execute("SELECT id, embedding FROM candidates ORDER BY id")]
        conn.execute("DROP TABLE candidate_notes")
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Migration must not re-embed resumes"))
    with TestClient(app) as restarted:
        assert restarted.get("/candidates/1").json()["notes"] == []
    with db.connect() as conn:
        assert [tuple(row) for row in conn.execute("SELECT id, embedding FROM candidates ORDER BY id")] == before


def test_notes_foreign_key_and_cascade(client):
    assert client.post("/candidates/1/notes", json={"note": "Existing note"}).status_code == 201
    with db.connect() as conn:
        conn.execute("DELETE FROM candidates WHERE id = 1")
    assert note_count() == 0
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_notes([note_row(999, "Orphan note")], np.stack([vector(1)]))


def test_database_rebuild_removes_notes_with_candidates(client):
    assert client.post("/candidates/1/notes", json={"note": "Existing note"}).status_code == 201
    assert build_database()["inserted"] == 3
    assert note_count() == 0
