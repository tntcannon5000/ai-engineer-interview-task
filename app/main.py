import sqlite3
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import FileResponse, JSONResponse
from google.genai.errors import APIError

from app import db, embeddings, extraction
from app.config import ROOT_DIR
from app.provider import provider_call
from app.schemas import (
    AddCandidateRequest,
    CandidateDetails,
    CandidateLookup,
    CandidateNote,
    NoteInput,
    SearchRequest,
    SearchResponse,
)
from app.taxonomy import ALL_SKILLS, SKILLS_TAXONOMY
from app.workflows import router as workflows_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Airport Applicant Semantic Search", lifespan=lifespan)


@app.exception_handler(db.DatabaseStateError)
async def database_error(request, exc):
    logger.warning("Database unavailable: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/", response_class=FileResponse, include_in_schema=False)
def recruiter_ui():
    return FileResponse(ROOT_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/taxonomy")
def taxonomy():
    return {"categories": SKILLS_TAXONOMY, "all_skills": ALL_SKILLS}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    from app.search import search as run_search
    return run_search(req)


@app.post("/candidates", response_model=CandidateDetails, status_code=201)
def add_candidate(req: AddCandidateRequest):
    db.assert_embedding_config()
    existing_id = db.find_exact_candidate(req.name, req.resume)
    if existing_id is not None:
        raise HTTPException(409, f"Candidate already exists (ID {existing_id}). Add notes to the existing candidate.")
    try:
        with provider_call("Candidate processing"):
            fields = req.fields if req.fields is not None else extraction.extract_fields(req.resume)
            vectors = embeddings.embed_documents([req.resume, *(note.note for note in req.notes)])
            db.validate_vectors(vectors, 1 + len(req.notes))
    except (RuntimeError, APIError) as exc:
        raise HTTPException(503, "Candidate processing service unavailable") from exc

    row = {"name": req.name.strip(), "resume": req.resume, **fields.model_dump()}
    try:
        saved = db.insert_candidate_with_notes(
            row, vectors[0], [note.model_dump(mode="json") for note in req.notes], vectors[1:]
        )
    except sqlite3.IntegrityError as exc:
        # The unique index also protects against simultaneous duplicate submissions.
        existing_id = db.find_exact_candidate(req.name, req.resume)
        if existing_id is not None:
            raise HTTPException(409, f"Candidate already exists (ID {existing_id}). Add notes to the existing candidate.") from exc
        raise HTTPException(409, "Initial notes contain an identical text and timestamp") from exc
    return CandidateDetails(**saved)


@app.post("/candidates/{candidate_id}/notes", response_model=CandidateNote, status_code=201)
def add_note(req: NoteInput, candidate_id: int = Path(gt=0)):
    db.assert_embedding_config()
    if db.get_candidate(candidate_id) is None:
        raise HTTPException(404, "candidate not found")
    try:
        with provider_call("Note embedding"):
            vectors = embeddings.embed_documents([req.note])
            db.validate_vectors(vectors, 1)
    except (RuntimeError, APIError) as exc:
        raise HTTPException(503, "Embedding service unavailable") from exc
    try:
        [saved] = db.insert_notes([{**req.model_dump(mode="json"), "candidate_id": candidate_id}], vectors)
    except sqlite3.IntegrityError as exc:
        if db.get_candidate(candidate_id) is None:
            raise HTTPException(404, "candidate not found") from exc
        raise HTTPException(409, "An identical note already exists for this candidate") from exc
    return CandidateNote(**saved)


@app.get("/candidates", response_model=list[CandidateLookup])
def lookup_candidates(name: str = Query(default="", max_length=200)):
    return db.lookup_candidates(name.strip())


@app.get("/candidates/{candidate_id}", response_model=CandidateDetails)
def get_candidate(candidate_id: int = Path(gt=0)):
    row = db.get_candidate(candidate_id)
    if row is None:
        raise HTTPException(404, "candidate not found")
    return CandidateDetails(**row)


app.include_router(workflows_router)
