"""Resume/query embeddings via the Gemini embedding API."""

import os

import numpy as np
from google import genai
from google.genai import types

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL
from app.provider import provider_call
from app.search_telemetry import record_call

_client: genai.Client | None = None
_BATCH = 50  # keep bulk requests below the observed free-tier per-minute limit


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=120_000,
                retry_options=types.HttpRetryOptions(
                    attempts=5, initial_delay=10, max_delay=60
                ),
            ),
        )
    return _client


@provider_call(f"Embeddings ({EMBEDDING_MODEL})")
def _embed(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        batch = texts[start : start + _BATCH]
        resp = _get_client().models.embed_content(
            model=EMBEDDING_MODEL,
            # Embedding 2 aggregates a plain list of strings into one embedding.
            # Explicit Content objects keep each candidate's vector separate.
            contents=[
                types.Content(parts=[types.Part.from_text(text=text)]) for text in batch
            ],
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        )
        if resp.embeddings is None or len(resp.embeddings) != len(batch):
            raise RuntimeError("Gemini must return one embedding per input document")
        counts = [getattr(getattr(e, "statistics", None), "token_count", None) for e in resp.embeddings]
        known = all(t is not None for t in counts)
        tokens = sum(counts) if known else sum(max(1, (len(t)+3)//4) for t in batch)
        record_call(EMBEDDING_MODEL, "embedding", tokens, basis="provider" if known else "estimated_chars_div_4")
        vectors.extend(e.values for e in resp.embeddings)
    if not vectors:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(texts), EMBEDDING_DIM) or not np.isfinite(matrix).all():
        raise RuntimeError("Gemini returned invalid embedding dimensions or values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Gemini returned a zero-length embedding")
    # Preserve the unit-vector contract used by SQLite storage and cosine ranking.
    return matrix / norms


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed resumes or notes for storage. Shape (n, EMBEDDING_DIM), L2-normalised."""
    return _embed([f"title: none | text: {text}" for text in texts])


def embed_query(text: str) -> np.ndarray:
    """Embed a search query. Shape (EMBEDDING_DIM,), L2-normalised."""
    return _embed([f"task: search result | query: {text}"])[0]


def embed_queries(texts: list[str]) -> np.ndarray:
    return _embed([f"task: search result | query: {text}" for text in texts])
