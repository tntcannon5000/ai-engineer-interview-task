from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import types

from app import embeddings
from app.config import EMBEDDING_DIM, EMBEDDING_MODEL


def stub_client(monkeypatch, responder):
    calls = []

    def embed_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(embeddings=responder(kwargs))

    client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    monkeypatch.setattr(embeddings, "_get_client", lambda: client)
    return calls


def test_documents_remain_separate_and_ordered_across_batches(monkeypatch):
    next_index = 0

    def respond(request):
        nonlocal next_index
        results = []
        for content in request["contents"]:
            assert isinstance(content, types.Content)
            assert content.parts[0].text == f"title: none | text: Resume {next_index}"
            vector = np.zeros(EMBEDDING_DIM)
            vector[next_index] = 2
            results.append(SimpleNamespace(values=vector.tolist()))
            next_index += 1
        return results

    calls = stub_client(monkeypatch, respond)
    matrix = embeddings.embed_documents([f"Resume {i}" for i in range(101)])

    assert [len(call["contents"]) for call in calls] == [50, 50, 1]
    for call in calls:
        assert call["model"] == EMBEDDING_MODEL == "gemini-embedding-2"
        assert call["config"].task_type is None
        assert call["config"].output_dimensionality == EMBEDDING_DIM
    assert matrix.shape == (101, EMBEDDING_DIM)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(np.argmax(matrix, axis=1), np.arange(101))
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1)


def test_query_uses_search_retrieval_instruction(monkeypatch):
    def respond(request):
        assert request["contents"][0].parts[0].text == (
            "task: search result | query: Aircraft mechanic"
        )
        return [SimpleNamespace(values=[1.0] * EMBEDDING_DIM)]

    calls = stub_client(monkeypatch, respond)
    vector = embeddings.embed_query("Aircraft mechanic")

    assert vector.shape == (EMBEDDING_DIM,)
    assert calls[0]["config"].task_type is None
    np.testing.assert_allclose(np.linalg.norm(vector), 1)


def test_empty_documents_skip_gemini(monkeypatch):
    calls = stub_client(monkeypatch, lambda request: pytest.fail("Unexpected API call"))
    matrix = embeddings.embed_documents([])
    assert matrix.shape == (0, EMBEDDING_DIM)
    assert not calls


def test_aggregated_response_is_rejected(monkeypatch):
    stub_client(monkeypatch, lambda request: [SimpleNamespace(values=[1.0] * EMBEDDING_DIM)])
    with pytest.raises(RuntimeError, match="invalid output") as error:
        embeddings.embed_documents(["Pilot resume", "Mechanic resume"])
    assert "one embedding per input" in str(error.value.__cause__)


@pytest.mark.parametrize(
    "values",
    [[1.0] * (EMBEDDING_DIM - 1), [float("nan")] * EMBEDDING_DIM, [0.0] * EMBEDDING_DIM],
)
def test_invalid_vectors_are_rejected(monkeypatch, values):
    stub_client(monkeypatch, lambda request: [SimpleNamespace(values=values)])
    with pytest.raises(RuntimeError):
        embeddings.embed_documents(["Pilot resume"])
