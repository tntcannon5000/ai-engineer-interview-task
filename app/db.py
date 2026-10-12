import json
import sqlite3
import hashlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from app.config import DB_PATH, EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_FORMAT_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    resume TEXT NOT NULL,
    skills TEXT NOT NULL,
    current_job_title TEXT NOT NULL,
    past_job_titles TEXT NOT NULL,
    years_experience INTEGER NOT NULL,
    location_city TEXT NOT NULL,
    location_region TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    note TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    embedding BLOB NOT NULL,
    UNIQUE (candidate_id, timestamp, note)
);
CREATE TABLE IF NOT EXISTS embedding_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS seed_candidates (
    seed_id INTEGER PRIMARY KEY,
    candidate_id INTEGER NOT NULL UNIQUE REFERENCES candidates(id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meeting_imports (
    meeting_id TEXT NOT NULL,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    note_id INTEGER NOT NULL REFERENCES candidate_notes(id) ON DELETE CASCADE,
    PRIMARY KEY (meeting_id, candidate_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS candidates_exact_identity ON candidates(name, resume);
"""

EMBEDDING_CONFIG = {"model": EMBEDDING_MODEL, "dimensions": str(EMBEDDING_DIM),
                    "format_version": EMBEDDING_FORMAT_VERSION}
CANDIDATE_COLUMNS = ("name", "resume", "skills", "current_job_title", "past_job_titles",
                     "years_experience", "location_city", "location_region")


class DatabaseStateError(RuntimeError):
    """Stored data cannot safely be used with the current configuration."""


def _set_embedding_config(conn):
    conn.executemany("INSERT OR REPLACE INTO embedding_metadata VALUES (?, ?)", EMBEDDING_CONFIG.items())


def assert_embedding_config(conn=None):
    if conn is None:
        with connect() as conn:
            return assert_embedding_config(conn)
    metadata = dict(conn.execute("SELECT key, value FROM embedding_metadata"))
    if metadata != EMBEDDING_CONFIG:
        raise DatabaseStateError("Embedding metadata is missing or incompatible. Use scripts/build_db.py --rebuild; "
                                 "for a verified compatible legacy DB, use --adopt-existing.")


def validate_vectors(matrix, count):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (count, EMBEDDING_DIM) or not np.isfinite(matrix).all():
        raise DatabaseStateError("Stored/generated embeddings have invalid dimensions or values")
    if count and not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
        raise DatabaseStateError("Stored/generated embeddings must be unit-length vectors")
    return matrix


def validate_stored_vectors(conn):
    for table in ("candidates", "candidate_notes"):
        blobs = [row[0] for row in conn.execute(f"SELECT embedding FROM {table}")]
        try:
            matrix = np.stack([from_blob(blob) for blob in blobs]) if blobs else np.zeros((0, EMBEDDING_DIM))
        except ValueError as exc:
            raise DatabaseStateError(f"Invalid embedding storage in {table}") from exc
        validate_vectors(matrix, len(blobs))


def candidate_fingerprint(row):
    fields = {key: row[key] for key in CANDIDATE_COLUMNS}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def state_token(conn=None):
    if conn is None:
        with connect() as conn:
            return state_token(conn)
    digest = hashlib.sha256()
    for table, order in (("candidates", "id"), ("candidate_notes", "id"),
                         ("embedding_metadata", "key"), ("seed_candidates", "seed_id"),
                         ("meeting_imports", "meeting_id, candidate_id")):
        for row in conn.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            digest.update(repr(tuple(row)).encode())
    return digest.hexdigest()

_db_path: Path = DB_PATH


def set_db_path(path: Path) -> None:
    global _db_path
    _db_path = path


@contextmanager
def connect():
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(reset: bool = False) -> None:
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if reset:
            conn.execute("DROP TABLE IF EXISTS meeting_imports")
            conn.execute("DROP TABLE IF EXISTS seed_candidates")
            conn.execute("DROP TABLE IF EXISTS candidate_notes")
            conn.execute("DROP TABLE IF EXISTS candidates")
            conn.execute("DROP TABLE IF EXISTS embedding_metadata")
        # executescript commits pending transactions; execute statements individually.
        for statement in SCHEMA.split(";"):
            if statement.strip():
                conn.execute(statement)
        if conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0:
            if not conn.execute("SELECT COUNT(*) FROM embedding_metadata").fetchone()[0]:
                _set_embedding_config(conn)


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _insert_candidate(conn: sqlite3.Connection, row: dict, vec: np.ndarray, candidate_id=None) -> int:
    cur = conn.execute(
        """INSERT INTO candidates
           (id, name, resume, skills, current_job_title, past_job_titles,
            years_experience, location_city, location_region, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            candidate_id,
            row["name"],
            row["resume"],
            json.dumps(row["skills"]),
            row["current_job_title"],
            json.dumps(row["past_job_titles"]),
            int(row["years_experience"]),
            row["location_city"],
            row["location_region"],
            to_blob(vec),
        ),
    )
    return cur.lastrowid


def insert_candidates(rows: list[dict], embeddings: np.ndarray) -> list[int]:
    embeddings = validate_vectors(embeddings, len(rows))
    with connect() as conn:
        assert_embedding_config(conn)
        return [
            _insert_candidate(conn, row, vec)
            for row, vec in zip(rows, embeddings, strict=True)
        ]


def _insert_notes(
    conn: sqlite3.Connection, rows: list[dict], vectors: np.ndarray, *, skip_duplicates: bool = False
) -> list[dict]:
    conflict = "ON CONFLICT(candidate_id, timestamp, note) DO NOTHING" if skip_duplicates else ""
    inserted = []
    for row, vector in zip(rows, vectors, strict=True):
        saved = conn.execute(
            f"""INSERT INTO candidate_notes (candidate_id, note, timestamp, embedding)
                VALUES (?, ?, ?, ?) {conflict}
                RETURNING id, candidate_id, note, timestamp""",
            (row["candidate_id"], row["note"], row["timestamp"], to_blob(vector)),
        ).fetchone()
        if saved is not None:
            inserted.append(dict(saved))
    return inserted


def insert_candidate_with_notes(
    row: dict, resume_vector: np.ndarray, notes: list[dict], note_vectors: np.ndarray
) -> dict:
    validate_vectors(np.asarray([resume_vector]), 1)
    validate_vectors(note_vectors, len(notes))
    with connect() as conn:
        assert_embedding_config(conn)
        candidate_id = _insert_candidate(conn, row, resume_vector)
        note_rows = [{**note, "candidate_id": candidate_id} for note in notes]
        saved_notes = _insert_notes(conn, note_rows, note_vectors)
    return {"id": candidate_id, **row, "notes": saved_notes}


def insert_notes(rows: list[dict], vectors: np.ndarray, *, skip_duplicates: bool = False,
                 expected_state=None) -> list[dict]:
    vectors = validate_vectors(vectors, len(rows))
    with connect() as conn:
        if expected_state is not None:
            conn.execute("BEGIN IMMEDIATE")
            if state_token(conn) != expected_state:
                raise DatabaseStateError("Database changed during note embedding; retry the import")
        assert_embedding_config(conn)
        return _insert_notes(conn, rows, vectors, skip_duplicates=skip_duplicates)


def get_candidate_ids() -> set[int]:
    with connect() as conn:
        return {row[0] for row in conn.execute("SELECT id FROM candidates")}


def find_exact_candidate(name: str, resume: str) -> int | None:
    with connect() as conn:
        row = conn.execute("SELECT id FROM candidates WHERE name = ? AND resume = ?",
                           (name, resume)).fetchone()
    return row[0] if row else None


def get_note_keys() -> set[tuple[int, str, str]]:
    with connect() as conn:
        return {
            tuple(row)
            for row in conn.execute("SELECT candidate_id, timestamp, note FROM candidate_notes")
        }


def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["skills"] = json.loads(d["skills"])
    d["past_job_titles"] = json.loads(d["past_job_titles"])
    d.pop("embedding", None)
    return d


def lookup_candidates(name: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT id, name, current_job_title, location_city, location_region
               FROM candidates WHERE instr(lower(name), lower(?)) > 0
               ORDER BY name COLLATE NOCASE, id""", (name,)).fetchall()
    return [dict(row) for row in rows]


def get_candidate(candidate_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if row is None:
            return None
        notes = conn.execute(
            """SELECT id, candidate_id, note, timestamp FROM candidate_notes
               WHERE candidate_id = ? ORDER BY id""",
            (candidate_id,),
        ).fetchall()
    # ISO timestamps with and without fractional seconds do not sort uniformly as text.
    ordered_notes = sorted(notes, key=lambda note: datetime.fromisoformat(note["timestamp"]))
    return {**row_to_dict(row), "notes": [dict(note) for note in ordered_notes]}


def query_notes(candidate_ids: list[int]) -> tuple[list[dict], np.ndarray]:
    if not candidate_ids:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    placeholders = ",".join("?" for _ in candidate_ids)
    with connect() as conn:
        assert_embedding_config(conn)
        notes = conn.execute(
            f"SELECT * FROM candidate_notes WHERE candidate_id IN ({placeholders}) ORDER BY id",
            candidate_ids,
        ).fetchall()
    if not notes:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    try:
        matrix = validate_vectors(np.stack([from_blob(note["embedding"]) for note in notes]), len(notes))
    except ValueError as exc:
        raise DatabaseStateError("Invalid stored note vectors") from exc
    metadata = [
        {key: value for key, value in dict(note).items() if key != "embedding"}
        for note in notes
    ]
    return metadata, matrix


def count_candidates() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]


def _like_pattern(value: str) -> str:
    """Case-insensitive 'includes' pattern; escapes LIKE wildcards in user input."""
    escaped = value.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def query_candidates(
    *,
    skills: list[str] | None = None,
    current_job_title: str | None = None,
    past_job_titles: list[str] | None = None,
    min_years_experience: int | None = None,
    max_years_experience: int | None = None,
    location_city: str | None = None,
    location_region: str | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Apply structured filters in SQL and return (rows, embedding matrix).

    Text filters are case-insensitive substring matches ("captain" matches "Senior Captain").
    Skills must all be present (exact, case-insensitive). Past titles: each value must be
    contained in at least one past title.
    """
    clauses: list[str] = []
    params: list = []

    for column, value in (
        ("current_job_title", current_job_title),
        ("location_city", location_city),
        ("location_region", location_region),
    ):
        if value:
            clauses.append(f"{column} LIKE ? ESCAPE '\\'")
            params.append(_like_pattern(value))

    if min_years_experience is not None:
        clauses.append("years_experience >= ?")
        params.append(min_years_experience)
    if max_years_experience is not None:
        clauses.append("years_experience <= ?")
        params.append(max_years_experience)

    # List columns are stored as JSON; require every requested value to be present.
    for v in skills or []:
        clauses.append(
            "EXISTS (SELECT 1 FROM json_each(candidates.skills) WHERE LOWER(json_each.value) = LOWER(?))"
        )
        params.append(v.strip())
    for v in past_job_titles or []:
        clauses.append(
            "EXISTS (SELECT 1 FROM json_each(candidates.past_job_titles) WHERE json_each.value LIKE ? ESCAPE '\\')"
        )
        params.append(_like_pattern(v))

    sql = "SELECT * FROM candidates"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id"

    with connect() as conn:
        assert_embedding_config(conn)
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return [], np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    try:
        matrix = validate_vectors(np.stack([from_blob(r["embedding"]) for r in rows]), len(rows))
    except ValueError as exc:
        raise DatabaseStateError("Invalid stored resume vectors") from exc
    return [row_to_dict(r) for r in rows], matrix


def pending_seeds(rows, conn):
    pending = []
    for row in rows:
        mapping = conn.execute("SELECT * FROM seed_candidates WHERE seed_id = ?", (row["id"],)).fetchone()
        if mapping is not None:
            stored = conn.execute("SELECT * FROM candidates WHERE id = ?", (mapping["candidate_id"],)).fetchone()
            if stored is None or mapping["fingerprint"] != candidate_fingerprint(row):
                raise DatabaseStateError(f"Seed {row['id']} changed; use an explicit rebuild")
            if candidate_fingerprint(row_to_dict(stored)) != mapping["fingerprint"]:
                raise DatabaseStateError(f"Stored seed {row['id']} differs from its source")
        elif conn.execute("SELECT 1 FROM candidates WHERE id = ?", (row["id"],)).fetchone():
            raise DatabaseStateError(f"Candidate ID {row['id']} is occupied without a verified seed mapping; "
                                     "use --adopt-existing for a compatible legacy DB or explicitly rebuild")
        else:
            pending.append(row)
    return pending


def write_seeds(rows, vectors, *, reset, expected_state):
    vectors = validate_vectors(vectors, len(rows))
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if state_token(conn) != expected_state:
            raise DatabaseStateError("Database changed during embedding; retry the build")
        if reset:
            conn.execute("DELETE FROM seed_candidates")
            conn.execute("DELETE FROM candidate_notes")
            conn.execute("DELETE FROM candidates")
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('candidates', 'candidate_notes')")
            conn.execute("DELETE FROM embedding_metadata")
            _set_embedding_config(conn)
        else:
            assert_embedding_config(conn)
            pending_seeds(rows, conn)
        for row, vector in zip(rows, vectors, strict=True):
            candidate_id = _insert_candidate(conn, row, vector, candidate_id=row["id"])
            conn.execute("INSERT INTO seed_candidates VALUES (?, ?, ?)",
                         (row["id"], candidate_id, candidate_fingerprint(row)))


def adopt_existing_seeds(rows):
    """Explicit operator assertion of legacy model/format, with identity/vector checks."""
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        metadata = dict(conn.execute("SELECT key, value FROM embedding_metadata"))
        if metadata and metadata != EMBEDDING_CONFIG:
            raise DatabaseStateError("Cannot adopt a database with conflicting embedding metadata; rebuild")
        validate_stored_vectors(conn)
        for row in rows:
            stored = conn.execute("SELECT * FROM candidates WHERE id = ?", (row["id"],)).fetchone()
            if stored is None or candidate_fingerprint(row_to_dict(stored)) != candidate_fingerprint(row):
                raise DatabaseStateError(f"Cannot verify legacy candidate {row['id']} against the seed source")
            mapping = conn.execute("SELECT * FROM seed_candidates WHERE seed_id = ?", (row["id"],)).fetchone()
            if mapping and (mapping["candidate_id"] != row["id"] or mapping["fingerprint"] != candidate_fingerprint(row)):
                raise DatabaseStateError(f"Conflicting seed mapping for {row['id']}")
            conn.execute("INSERT OR IGNORE INTO seed_candidates VALUES (?, ?, ?)",
                         (row["id"], row["id"], candidate_fingerprint(row)))
        if conn.execute("PRAGMA foreign_key_check").fetchall():
            raise DatabaseStateError("Database contains invalid candidate associations")
        _set_embedding_config(conn)


def get_meeting_note(meeting_id, candidate_id):
    with connect() as conn:
        row = conn.execute("""SELECT n.id, n.candidate_id, n.note, n.timestamp
            FROM meeting_imports m JOIN candidate_notes n ON n.id=m.note_id
            WHERE m.meeting_id=? AND m.candidate_id=?""", (meeting_id, candidate_id)).fetchone()
    return dict(row) if row else None


def insert_meeting_note(meeting_id, row, vectors):
    vectors = validate_vectors(vectors, 1)
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert_embedding_config(conn)
        existing = conn.execute("SELECT note_id FROM meeting_imports WHERE meeting_id=? AND candidate_id=?",
                                (meeting_id, row["candidate_id"])).fetchone()
        if existing:
            saved = conn.execute("SELECT id, candidate_id, note, timestamp FROM candidate_notes WHERE id=?", (existing[0],)).fetchone()
            return dict(saved), False
        [saved] = _insert_notes(conn, [row], vectors)
        conn.execute("INSERT INTO meeting_imports VALUES (?, ?, ?)", (meeting_id, row["candidate_id"], saved["id"]))
        return saved, True
