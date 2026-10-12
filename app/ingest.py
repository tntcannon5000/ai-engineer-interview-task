"""Validated, repeatable seed imports and explicit legacy database adoption."""

import json
import logging
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from app import db, embeddings
from app.config import NOTES_PATH, RESUMES_PATH
from app.schemas import ImportNote, SeedCandidate

logger = logging.getLogger(__name__)


def read_records(path, model):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid UTF-8 JSON") from exc
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    records = []
    for index, row in enumerate(data, 1):
        try:
            records.append(model.model_validate(row))
        except ValidationError as exc:
            errors = "; ".join(f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                               for error in exc.errors(include_input=False))
            raise ValueError(f"{path}: row {index}: {errors}") from exc
    return records


def read_seeds(path=None):
    records = read_records(path or RESUMES_PATH, SeedCandidate)
    if not records:
        raise ValueError("Resume source must contain at least one candidate")
    identities = [(row.name, row.resume) for row in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Resume source contains exact duplicate candidates")
    # Supplied notes reference the original resume list by one-based position.
    # Keep this fixture association internal; fingerprints verify replay safety.
    return [{"id": index, **row.model_dump()} for index, row in enumerate(records, 1)]


def build_database(reset=True, path=None):
    rows = read_seeds(path)
    db.init_db()
    with db.connect() as conn:
        if not reset:
            db.assert_embedding_config(conn)
            db.validate_stored_vectors(conn)
            pending = db.pending_seeds(rows, conn)
        else:
            pending = rows
        expected_state = db.state_token(conn)
    if not pending:
        return {"inserted": 0, "total": db.count_candidates()}
    # All validation and provider work precedes the one replacement transaction.
    vectors = embeddings.embed_documents([row["resume"] for row in pending])
    db.write_seeds(pending, vectors, reset=reset, expected_state=expected_state)
    return {"inserted": len(pending), "total": db.count_candidates()}


def check_database(path=None):
    """Read-only setup check: file existence alone is insufficient."""
    rows = read_seeds(path)
    if not db._db_path.exists():
        raise db.DatabaseStateError("Database is missing; run setup")
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db._db_path.resolve().as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        db.assert_embedding_config(conn)
        db.validate_stored_vectors(conn)
        if db.pending_seeds(rows, conn):
            raise db.DatabaseStateError("Database is incomplete; run setup to import missing seeds")
        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise db.DatabaseStateError("Database contains invalid candidate associations")
    except sqlite3.DatabaseError as exc:
        raise db.DatabaseStateError("Database schema is missing or unreadable; run setup or explicitly rebuild") from exc
    finally:
        if conn is not None:
            conn.close()


def adopt_existing(path=None):
    rows = read_seeds(path)
    db.init_db()
    db.adopt_existing_seeds(rows)
    return {"adopted": len(rows), "total": db.count_candidates()}


def note_quality_counts(notes):
    """Diagnostics only: keep raw notes and do not alter ranking in phase 2."""
    counts = {"placeholder": 0, "numeric_only": 0, "other_text": 0}
    for note in notes:
        text = note.note.strip()
        kind = "placeholder" if text.casefold() in {"-", "n/a", "na", "none"} else (
            "numeric_only" if text.isdigit() else "other_text")
        counts[kind] += 1
    return counts


def import_notes(path=None, *, seed_ids=None, seed_source=None):
    path = Path(path or NOTES_PATH)
    notes = read_records(path, ImportNote)
    logger.info("Note import input quality: %s", note_quality_counts(notes))
    db.init_db()
    db.assert_embedding_config()
    expected_state = db.state_token()
    if seed_ids is None:
        seed_ids = path.resolve() == NOTES_PATH.resolve()
    mappings = {}
    if seed_ids:
        # Shipped notes use seed IDs; arbitrary files use database IDs by default.
        rows = read_seeds(seed_source)
        with db.connect() as conn:
            if db.pending_seeds(rows, conn):
                raise ValueError("Seed mapping is incomplete; import candidates before supplied notes")
            mappings = {row[0]: row[1] for row in conn.execute("SELECT seed_id, candidate_id FROM seed_candidates")}
        missing = {note.candidate_id for note in notes} - mappings.keys()
    else:
        missing = {note.candidate_id for note in notes} - db.get_candidate_ids()
    if missing:
        raise ValueError(f"Unknown candidate IDs: {sorted(missing)}. Build the candidates database first.")

    seen = db.get_note_keys()
    pending = []
    for note in notes:
        row = note.model_dump(mode="json")
        if seed_ids:
            row["candidate_id"] = mappings[row["candidate_id"]]
        key = (row["candidate_id"], row["timestamp"], row["note"])
        if key not in seen:
            pending.append(row)
            seen.add(key)
    if not pending:
        return {"inserted": 0, "skipped": len(notes)}

    vectors = embeddings.embed_documents([row["note"] for row in pending])
    saved = db.insert_notes(pending, vectors, skip_duplicates=True, expected_state=expected_state)
    return {"inserted": len(saved), "skipped": len(notes) - len(saved)}
