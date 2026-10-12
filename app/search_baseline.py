from app import db, embeddings, search_telemetry as telemetry
from app.provider import provider_call
from app.schemas import SearchResponse, SearchResult
from fastapi import HTTPException
from google.genai.errors import APIError
import numpy as np

def baseline(req):
    rows, matrix = db.query_candidates(
        skills=req.skills,
        current_job_title=req.current_job_title,
        past_job_titles=req.past_job_titles,
        min_years_experience=req.min_years_experience,
        max_years_experience=req.max_years_experience,
        location_city=req.location_city,
        location_region=req.location_region,
    )
    if not rows:
        return SearchResponse(results=[], total_matching_filters=0)

    notes, note_matrix = db.query_notes([row["id"] for row in rows])
    try:
        with provider_call("Search query embedding"):
            query_vec = embeddings.embed_query(req.query)
            db.validate_vectors(np.asarray([query_vec]), 1)
    except (RuntimeError, APIError) as exc:
        raise HTTPException(503, "Embedding service unavailable") from exc
    scores = matrix @ query_vec  # embeddings are L2-normalised, so dot == cosine
    matched_notes = [None] * len(rows)
    candidate_indices = {row["id"]: i for i, row in enumerate(rows)}
    for note, score in zip(notes, note_matrix @ query_vec, strict=True):
        index = candidate_indices[note["candidate_id"]]
        # Strict comparison preserves the resume on a tie, then the first note.
        if score > scores[index]:
            scores[index] = score
            matched_notes[index] = note
    telemetry.shortlist([rows[i]["id"] for i in np.argsort(-scores, kind="stable")[:12]])
    order = np.argsort(-scores, kind="stable")[: req.top_k]
    results = [
        SearchResult(
            **rows[i], score=float(scores[i]),
            match_source="note" if matched_notes[i] is not None else "resume",
            matched_note=matched_notes[i],
        )
        for i in order
    ]
    return SearchResponse(results=results, total_matching_filters=len(rows))
