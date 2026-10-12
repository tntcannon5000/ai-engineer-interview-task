"""Draft preparation and explicit, replay-safe simulated meeting ingestion."""
import sqlite3
from time import perf_counter
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import Field, field_validator
from starlette.concurrency import run_in_threadpool
from app import db, documents, preextract, granola, embeddings
from app.provider import provider_call, ProcessingError
from app.schemas import StrictInput, CandidateNote, NoteInput

router = APIRouter()


class DocumentText(StrictInput):
    filename: str
    text: str
    extraction_ms: float


class PreextractRequest(StrictInput):
    text: str = Field(min_length=1, max_length=documents.MAX_CHARS)

    @field_validator("text")
    @classmethod
    def non_blank(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value


class DraftResponse(StrictInput):
    draft: preextract.CandidateDraft
    extraction_ms: float


class MeetingImport(StrictInput):
    candidate_id: int = Field(gt=0)
    note: str = Field(min_length=1, max_length=documents.MAX_CHARS)

    @field_validator("note")
    @classmethod
    def non_blank(cls, value):
        return NoteInput.non_blank_note(value)


class MeetingImportResult(StrictInput):
    note: CandidateNote
    inserted: bool


@router.post("/documents/extract", response_model=DocumentText)
async def extract_document(file: UploadFile = File(...)):
    start = perf_counter()
    try:
        content = await file.read(documents.MAX_BYTES + 1)
        if len(content) > documents.MAX_BYTES:
            raise HTTPException(413, "File exceeds 10 MB")
        text = await run_in_threadpool(documents.extract_text, file.filename or "", content)
        return DocumentText(filename=file.filename or "", text=text, extraction_ms=(perf_counter()-start)*1000)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()


@router.post("/candidates/pre-extract", response_model=DraftResponse)
def pre_extract(req: PreextractRequest):
    start = perf_counter()
    try:
        draft = preextract.extract_draft(req.text)
    except ProcessingError as exc:
        raise HTTPException(503, "Pre-extraction unavailable; your text has been kept. Try again or fill in fields manually.") from exc
    return DraftResponse(draft=draft, extraction_ms=(perf_counter()-start)*1000)


@router.get("/integrations/granola/meetings", response_model=list[granola.Meeting])
def meetings():
    return granola.list_meetings()


@router.post("/integrations/granola/meetings/{meeting_id}/import", response_model=MeetingImportResult)
def import_meeting(meeting_id: str, req: MeetingImport):
    meeting = granola.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(404, "Meeting not found")
    db.assert_embedding_config()
    if db.get_candidate(req.candidate_id) is None:
        raise HTTPException(404, "Candidate not found")
    existing = db.get_meeting_note(meeting_id, req.candidate_id)
    if existing:
        return MeetingImportResult(note=existing, inserted=False)
    note = NoteInput(note=f"Granola · {meeting.title} [{meeting.id}]\n\n{req.note}", timestamp=meeting.timestamp)
    try:
        with provider_call("Meeting embedding"):
            vectors = embeddings.embed_documents([note.note])
        saved, inserted = db.insert_meeting_note(meeting_id,
            {"candidate_id": req.candidate_id, **note.model_dump(mode="json")}, vectors)
    except ProcessingError as exc:
        raise HTTPException(503, "Meeting processing unavailable; nothing was saved") from exc
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "Candidate or note changed; refresh and try again") from exc
    return MeetingImportResult(note=saved, inserted=inserted)
