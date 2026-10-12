"""Regression checks for destructive operations, identities and invalid inputs."""

import json
import sqlite3
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from app import db, embeddings, extraction, ingest
from app.schemas import ImportNote
from tests.test_api import client, fake_embed


def seeds():
    return json.loads(ingest.RESUMES_PATH.read_text())


def write_seeds(rows):
    ingest.RESUMES_PATH.write_text(json.dumps(rows))


def test_database_admin_not_exposed(client):
    assert client.post("/db/create", json={"reset": True}).status_code == 404
    assert "/db/create" not in client.get("/openapi.json").json()["paths"]
    assert db.count_candidates() == 3


def test_exact_candidate_rejected_before_provider_without_saving_notes(client, monkeypatch):
    candidate = db.get_candidate(1)
    monkeypatch.setattr(extraction, "extract_fields", lambda _: pytest.fail("Duplicate called provider"))
    response = client.post("/candidates", json={"name": f" {candidate['name']} ",
                           "resume": candidate["resume"], "notes": [{"note": "Do not silently save"}]})
    assert response.status_code == 409
    assert "ID 1" in response.json()["detail"]
    assert db.count_candidates() == 3
    assert db.get_candidate(1)["notes"] == []


def test_unique_index_prevents_duplicate_when_early_check_misses(client, monkeypatch):
    candidate = db.get_candidate(1)
    original = db.find_exact_candidate
    calls = 0
    def lookup(name, resume):
        nonlocal calls
        calls += 1
        return None if calls == 1 else original(name, resume)
    monkeypatch.setattr(db, "find_exact_candidate", lookup)
    from app.schemas import CandidateFields
    monkeypatch.setattr(extraction, "extract_fields", lambda _: CandidateFields.model_validate(candidate))
    response = client.post("/candidates", json={"name": candidate["name"], "resume": candidate["resume"]})
    assert response.status_code == 409
    assert "ID 1" in response.json()["detail"]
    assert db.count_candidates() == 3


@pytest.mark.parametrize("changed_field", ["name", "resume"])
def test_nonidentical_candidates_are_not_merged(client, monkeypatch, changed_field):
    candidate = db.get_candidate(1)
    from app.schemas import CandidateFields
    monkeypatch.setattr(extraction, "extract_fields", lambda _: CandidateFields.model_validate(candidate))
    payload = {"name": candidate["name"], "resume": candidate["resume"]}
    payload[changed_field] += " changed"
    assert client.post("/candidates", json=payload).status_code == 201
    assert db.count_candidates() == 4


@pytest.mark.parametrize("payload", [
    {"query": "  "}, {"query": "pilot", "location_regoin": "Texas"},
    {"query": "pilot", "location_city": "  "}, {"query": "pilot", "skills": [" "]},
    {"query": "pilot", "skills": ["made-up skill"]},
    {"query": "pilot", "past_job_titles": [""]},
    {"query": "pilot", "min_years_experience": 9, "max_years_experience": 2},
])
def test_invalid_search_does_not_call_provider(client, monkeypatch, payload):
    monkeypatch.setattr(embeddings, "embed_query", lambda _: pytest.fail("Invalid input called provider"))
    assert client.post("/search", json=payload).status_code == 422


@pytest.mark.parametrize("payload", [
    {"name": " ", "resume": "pilot"}, {"name": "Name", "resume": " \n"},
    {"name": "Name", "resume": "pilot", "resmue": "typo"},
    {"name": "Name", "resume": "pilot", "notes": [{"note": "text", "timstamp": "typo"}]},
])
def test_invalid_candidate_does_not_call_provider(client, monkeypatch, payload):
    monkeypatch.setattr(extraction, "extract_fields", lambda _: pytest.fail("Invalid input called provider"))
    assert client.post("/candidates", json=payload).status_code == 422
    assert db.count_candidates() == 3


def test_trim_and_canonicalise_search_and_positive_paths(client):
    response = client.post("/search", json={"query": "  pilot  ", "skills": [" atp certificate ", "ATP Certificate"]})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["results"]] == [1]
    assert client.get("/candidates/0").status_code == 422
    assert client.post("/candidates/-1/notes", json={"note": "text"}).status_code == 422


def test_rebuild_provider_failure_preserves_all_data(client, monkeypatch):
    client.post("/candidates/1/notes", json={"note": "Keep this note"})
    before = db.state_token()
    def fail(_):
        raise RuntimeError("provider failure")
    monkeypatch.setattr(embeddings, "embed_documents", fail)
    with pytest.raises(RuntimeError):
        ingest.build_database()
    assert db.state_token() == before


def test_rebuild_insert_failure_rolls_back_deletes_and_metadata(client, monkeypatch):
    client.post("/candidates/1/notes", json={"note": "Keep this note"})
    before = db.state_token()
    original = db._insert_candidate
    def insert(conn, row, vector, candidate_id=None):
        if candidate_id == 2:
            raise sqlite3.IntegrityError("simulated mid-build write failure")
        return original(conn, row, vector, candidate_id)
    monkeypatch.setattr(db, "_insert_candidate", insert)
    with pytest.raises(sqlite3.IntegrityError):
        ingest.build_database()
    assert db.state_token() == before


def test_build_detects_concurrent_change_instead_of_overwriting_it(client, monkeypatch):
    def embed(texts):
        with db.connect() as conn:
            conn.execute("UPDATE candidates SET name = 'Concurrent edit' WHERE id = 1")
        return fake_embed(texts)
    monkeypatch.setattr(embeddings, "embed_documents", embed)
    with pytest.raises(db.DatabaseStateError, match="changed during"):
        ingest.build_database()
    assert db.get_candidate(1)["name"] == "Concurrent edit"
    assert db.count_candidates() == 3


def test_reordered_seed_replay_is_rejected_before_embedding(client, monkeypatch):
    before = db.state_token()
    write_seeds(list(reversed(seeds())))
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Replay called provider"))
    with pytest.raises(db.DatabaseStateError, match="changed"):
        ingest.build_database(reset=False)
    assert db.state_token() == before


def test_incomplete_database_is_repaired_without_losing_other_data(client):
    with db.connect() as conn:
        conn.execute("DELETE FROM candidates WHERE id = 2")
    with pytest.raises(db.DatabaseStateError, match="incomplete"):
        ingest.check_database()
    assert ingest.build_database(reset=False) == {"inserted": 1, "total": 3}
    ingest.check_database()
    assert db.get_candidate(1)["name"] == "Pilot Person"


def test_ambiguous_seed_collision_rejected_before_provider(client, monkeypatch):
    with db.connect() as conn:
        conn.execute("DELETE FROM seed_candidates WHERE seed_id = 1")
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Collision called provider"))
    with pytest.raises(db.DatabaseStateError, match="occupied"):
        ingest.build_database(reset=False)


def test_changed_seed_rejected_before_importing_supplied_notes(client, tmp_path, monkeypatch):
    rows = seeds()
    rows[0]["name"] = "Different Person"
    write_seeds(rows)
    notes = tmp_path / "supplied.json"
    notes.write_text(json.dumps([{"candidate_id": 1, "note": "text", "timestamp": "2026-01-01T00:00:00Z"}]))
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Bad mapping called provider"))
    with pytest.raises(db.DatabaseStateError, match="changed"):
        ingest.import_notes(notes, seed_ids=True)


def test_seed_note_mapping_and_custom_database_ids_are_distinct(client, tmp_path):
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps([{"candidate_id": 1, "note": "Available weekends", "timestamp": "2026-01-01T00:00:00Z"}]))
    assert ingest.import_notes(notes, seed_ids=True) == {"inserted": 1, "skipped": 0}
    assert ingest.import_notes(notes, seed_ids=False) == {"inserted": 0, "skipped": 1}


@pytest.mark.parametrize("change", ["duplicate_candidate", "missing_name", "blank_resume", "bad_years", "extra_field"])
def test_bad_resume_source_fails_before_provider_and_preserves_data(client, monkeypatch, change):
    rows = seeds()
    if change == "duplicate_candidate": rows[1] = dict(rows[0])
    elif change == "missing_name": del rows[0]["name"]
    elif change == "blank_resume": rows[0]["resume"] = " "
    elif change == "bad_years": rows[0]["years_experience"] = -1
    else: rows[0]["unexpected"] = "typo"
    write_seeds(rows)
    before = db.state_token()
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Invalid source called provider"))
    with pytest.raises(ValueError):
        ingest.build_database()
    assert db.state_token() == before


def test_normalisation_is_shared_by_source_and_model_output(client):
    rows = seeds()
    rows[0].update(skills=[" atp certificate ", "ATP Certificate", "bogus"],
                   location_region=" tx ", current_job_title=" Captain ",
                   past_job_titles=[" First Officer ", "", "First Officer"])
    write_seeds(rows)
    ingest.build_database()
    candidate = db.get_candidate(1)
    assert candidate["skills"] == ["ATP Certificate"]
    assert candidate["location_region"] == "Texas"
    assert candidate["current_job_title"] == "Captain"
    assert candidate["past_job_titles"] == ["First Officer"]


@pytest.mark.parametrize("fault", ["missing", "model", "dimensions", "format_version", "corrupt_vector"])
def test_incompatible_embeddings_fail_before_provider(client, monkeypatch, fault):
    with db.connect() as conn:
        if fault == "missing": conn.execute("DELETE FROM embedding_metadata")
        elif fault == "corrupt_vector": conn.execute("UPDATE candidates SET embedding = ? WHERE id = 1", (b"bad",))
        else: conn.execute("UPDATE embedding_metadata SET value = 'wrong' WHERE key = ?", (fault,))
    monkeypatch.setattr(embeddings, "embed_query", lambda _: pytest.fail("Incompatible DB called provider"))
    monkeypatch.setattr(extraction, "extract_fields", lambda _: pytest.fail("Incompatible DB called provider"))
    assert client.post("/search", json={"query": "pilot"}).status_code == 503
    if fault != "corrupt_vector":
        assert client.post("/candidates", json={"name": "Name", "resume": "Resume"}).status_code == 503
        assert client.post("/candidates/1/notes", json={"note": "text"}).status_code == 503
    with pytest.raises(db.DatabaseStateError): ingest.check_database()


def test_verified_legacy_adoption_preserves_candidate_and_note_rows(client):
    client.post("/candidates/1/notes", json={"note": "Preserve raw text"})
    with db.connect() as conn:
        before = [tuple(row) for row in conn.execute("SELECT * FROM candidates ORDER BY id")]
        notes = [tuple(row) for row in conn.execute("SELECT * FROM candidate_notes ORDER BY id")]
        conn.execute("DELETE FROM embedding_metadata")
        conn.execute("DELETE FROM seed_candidates")
    assert ingest.adopt_existing()["adopted"] == 3
    ingest.check_database()
    with db.connect() as conn:
        assert before == [tuple(row) for row in conn.execute("SELECT * FROM candidates ORDER BY id")]
        assert notes == [tuple(row) for row in conn.execute("SELECT * FROM candidate_notes ORDER BY id")]


def test_failed_adoption_is_atomic(client):
    with db.connect() as conn:
        conn.execute("DELETE FROM embedding_metadata")
        conn.execute("DELETE FROM seed_candidates")
        conn.execute("UPDATE candidates SET name = 'Wrong person' WHERE id = 2")
    before = db.state_token()
    with pytest.raises(db.DatabaseStateError, match="candidate 2"):
        ingest.adopt_existing()
    assert db.state_token() == before


@pytest.mark.parametrize("response", [
    SimpleNamespace(parsed=None, text=None), SimpleNamespace(parsed=None, text="not JSON"),
    SimpleNamespace(parsed=None, text='{"years_experience": -1}'),
])
def test_invalid_extraction_output_is_503_and_leaves_no_writes(client, monkeypatch, response):
    monkeypatch.setattr(extraction, "get_client", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response)))
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Invalid extraction proceeded to embed"))
    before = db.state_token()
    assert client.post("/candidates", json={"name": "Name", "resume": "Resume"}).status_code == 503
    assert db.state_token() == before


def test_transport_failure_is_logged_without_document_or_secret(client, monkeypatch, caplog):
    def fail(**kwargs): raise httpx.ReadTimeout("secret-key and confidential document")
    monkeypatch.setattr(extraction, "get_client", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=fail)))
    assert client.post("/candidates", json={"name": "Name", "resume": "confidential document"}).status_code == 503
    assert "ReadTimeout" in caplog.text
    assert "secret-key" not in caplog.text and "confidential document" not in caplog.text


def test_messy_notes_preserved_and_audited_without_interpreting_them(client, tmp_path):
    texts = ["-", " n/a ", "72929905435", "Called, went to voicemail",
             "Recruiter: Nights?\nCandidate: Cannot work overnight.", "Available weekends"]
    rows = [{"candidate_id": 1, "note": text, "timestamp": "2026-01-01T00:00:00Z"} for text in texts]
    path = tmp_path / "messy.json"
    path.write_text(json.dumps(rows))
    assert ingest.note_quality_counts([ImportNote.model_validate(row) for row in rows]) == {
        "placeholder": 2, "numeric_only": 1, "other_text": 3}
    ingest.import_notes(path)
    assert [note["note"] for note in db.get_candidate(1)["notes"]] == texts


def test_malformed_note_file_reports_row_and_preserves_data(client, tmp_path, monkeypatch):
    rows = [{"candidate_id": 1, "note": "valid", "timestamp": "2026-01-01T00:00:00Z"},
            {"candidate_id": 2, "note": "valid", "timestamp": "bad"}]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(rows))
    before = db.state_token()
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Bad file called provider"))
    with pytest.raises(ValueError, match="row 2: timestamp"):
        ingest.import_notes(path)
    assert db.state_token() == before


def test_fresh_setup_and_replay_use_common_pipeline(client, tmp_path, monkeypatch):
    from scripts.setup import main
    monkeypatch.setattr(db, "_db_path", tmp_path / "fresh.db")
    notes = tmp_path / "supplied.json"
    notes.write_text(json.dumps([{"candidate_id": 1, "note": "Available weekends", "timestamp": "2026-01-01T00:00:00Z"}]))
    monkeypatch.setattr(ingest, "NOTES_PATH", notes)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    assert main(["--no-run"]) == 0
    assert db.count_candidates() == 3
    assert len(db.get_candidate(1)["notes"]) == 1
    before = db.state_token()
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Setup replay called provider"))
    assert main(["--no-run"]) == 0
    assert db.state_token() == before


@pytest.mark.parametrize("contents", ["{invalid JSON", "{}", b"\xff"])
def test_unreadable_source_is_rejected_without_writes(client, monkeypatch, contents):
    path = ingest.RESUMES_PATH
    path.write_bytes(contents if isinstance(contents, bytes) else contents.encode())
    before = db.state_token()
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Invalid source called provider"))
    with pytest.raises(ValueError): ingest.build_database()
    assert db.state_token() == before


def test_note_import_detects_database_change_during_provider_call(client, tmp_path, monkeypatch):
    path = tmp_path / "notes.json"
    path.write_text(json.dumps([{"candidate_id": 1, "note": "new note", "timestamp": "2026-01-01T00:00:00Z"}]))
    def embed(texts):
        with db.connect() as conn:
            conn.execute("UPDATE candidates SET name = 'Concurrent edit' WHERE id = 1")
        return fake_embed(texts)
    monkeypatch.setattr(embeddings, "embed_documents", embed)
    with pytest.raises(db.DatabaseStateError, match="changed during note embedding"):
        ingest.import_notes(path)
    assert db.get_candidate(1)["notes"] == []


def test_valid_model_output_uses_same_normalisation_as_seed_import(client, monkeypatch):
    fields = {key: value for key, value in seeds()[0].items() if key not in {"id", "name", "resume"}}
    fields.update(skills=[" atp certificate ", "bogus", "ATP Certificate"], location_region=" tx ")
    response = SimpleNamespace(parsed=None, text=json.dumps(fields))
    monkeypatch.setattr(extraction, "get_client", lambda: SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: response)))
    result = client.post("/candidates", json={"name": " Name ", "resume": " raw resume \n"})
    assert result.status_code == 201
    assert result.json()["name"] == "Name"
    assert result.json()["resume"] == " raw resume \n"
    assert result.json()["skills"] == ["ATP Certificate"]
    assert result.json()["location_region"] == "Texas"


def test_read_only_check_does_not_create_missing_database(client, tmp_path, monkeypatch):
    path = tmp_path / "missing.db"
    monkeypatch.setattr(db, "_db_path", path)
    with pytest.raises(db.DatabaseStateError, match="missing"):
        ingest.check_database()
    assert not path.exists()


def test_original_sample_import_and_note_associations(tmp_path, monkeypatch):
    from app.config import RESUMES_PATH, NOTES_PATH

    source = json.loads(RESUMES_PATH.read_text(encoding="utf-8"))
    assert all("id" not in row for row in source)
    monkeypatch.setattr(db, "_db_path", tmp_path / "samples.db")
    monkeypatch.setattr(embeddings, "embed_documents", fake_embed)
    assert ingest.build_database(reset=False) == {"inserted": 100, "total": 100}
    assert ingest.import_notes() == {"inserted": 119, "skipped": 0}
    expected = ingest.read_records(NOTES_PATH, ImportNote)
    with db.connect() as conn:
        actual = conn.execute("SELECT c.name, c.resume, n.note, n.timestamp FROM candidate_notes n JOIN candidates c ON c.id = n.candidate_id").fetchall()
    assert sorted(tuple(row) for row in actual) == sorted(
        (source[n.candidate_id - 1]["name"].strip(), source[n.candidate_id - 1]["resume"], n.note, n.model_dump(mode="json")["timestamp"])
        for n in expected
    )
    before = db.state_token()
    monkeypatch.setattr(embeddings, "embed_documents", lambda _: pytest.fail("Replay called provider"))
    assert ingest.build_database(reset=False) == {"inserted": 0, "total": 100}
    assert ingest.import_notes() == {"inserted": 0, "skipped": 119}
    assert db.state_token() == before
