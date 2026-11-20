import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from verilegal.config import Settings
from verilegal.indexing import sparse


def raw_vector(indices: list[int], values: list[float]) -> SimpleNamespace:
    return SimpleNamespace(indices=np.asarray(indices), values=np.asarray(values))


class FakeModel:
    def __init__(self) -> None:
        self.document_calls: list[tuple[list[str], int]] = []
        self.query_calls: list[str] = []
        self.document_result: list[SimpleNamespace] | None = None
        self.query_result = [raw_vector([9, 3], [1.0, 1.0])]

    def embed(self, documents: list[str], *, batch_size: int) -> Iterator[SimpleNamespace]:
        self.document_calls.append((list(documents), batch_size))
        if self.document_result is not None:
            yield from self.document_result
            return
        for index, _ in enumerate(documents):
            yield raw_vector([9, 3], [2.5 + index, 1.5 + index])

    def query_embed(self, text: str) -> Iterator[SimpleNamespace]:
        self.query_calls.append(text)
        yield from self.query_result


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


@pytest.fixture(autouse=True)
def isolate_factory() -> Iterator[None]:
    sparse.get_sparse_encoder.cache_clear()
    yield
    sparse.get_sparse_encoder.cache_clear()


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeModel, list[Settings]]:
    model = FakeModel()
    loads: list[Settings] = []

    def load(configuration: Settings) -> FakeModel:
        loads.append(configuration)
        return model

    monkeypatch.setattr(sparse, "_load_model", load)
    return model, loads


def test_documents_keep_weights_and_alignment_while_sorting_indices(
    backend: tuple[FakeModel, list[Settings]],
) -> None:
    model, loads = backend
    encoder = sparse.Bm25Encoder(settings(sparse_batch_size=8))
    assert loads == []
    texts = ["Chiffrement des données", "Résilience opérationnelle"]
    vectors = encoder.embed_documents(texts)
    assert vectors == [
        sparse.SparseVector((3, 9), (1.5, 2.5)),
        sparse.SparseVector((3, 9), (2.5, 3.5)),
    ]
    assert model.document_calls == [(texts, 8)]
    assert len(loads) == 1


def test_queries_use_query_weights_and_share_the_loaded_model(
    backend: tuple[FakeModel, list[Settings]],
) -> None:
    model, loads = backend
    encoder = sparse.Bm25Encoder(settings())
    assert encoder.embed_query("Quels droits ?") == sparse.SparseVector((3, 9), (1.0, 1.0))
    assert model.query_calls == ["Quels droits ?"]
    assert model.document_calls == []
    encoder.embed_documents(["Un passage"])
    assert len(loads) == 1


def test_empty_batch_does_not_load_model(backend: tuple[FakeModel, list[Settings]]) -> None:
    _, loads = backend
    assert sparse.Bm25Encoder(settings()).embed_documents([]) == []
    assert loads == []


@pytest.mark.parametrize("text", ["", " \n"])
def test_blank_query_rejected_without_loading(
    backend: tuple[FakeModel, list[Settings]], text: str
) -> None:
    _, loads = backend
    with pytest.raises(ValueError, match="blank"):
        sparse.Bm25Encoder(settings()).embed_query(text)
    assert loads == []


@pytest.mark.parametrize("texts", ["not a batch", ["text", None], [42]])
def test_invalid_batch_types_rejected(
    backend: tuple[FakeModel, list[Settings]], texts: Any
) -> None:
    _, loads = backend
    with pytest.raises(TypeError):
        sparse.Bm25Encoder(settings()).embed_documents(texts)
    assert loads == []


def test_zero_term_vector_is_valid_for_stopword_only_query(
    backend: tuple[FakeModel, list[Settings]],
) -> None:
    model, _ = backend
    model.query_result = [raw_vector([], [])]
    assert sparse.Bm25Encoder(settings()).embed_query("le la les") == sparse.SparseVector((), ())


@pytest.mark.parametrize("count", [0, 2])
def test_query_requires_exactly_one_backend_result(
    backend: tuple[FakeModel, list[Settings]], count: int
) -> None:
    model, _ = backend
    model.query_result = [raw_vector([1], [1.0])] * count
    with pytest.raises(ValueError, match="one vector per query"):
        sparse.Bm25Encoder(settings()).embed_query("question")


@pytest.mark.parametrize("count", [0, 2])
def test_document_vector_count_must_match_input(
    backend: tuple[FakeModel, list[Settings]], count: int
) -> None:
    model, _ = backend
    model.document_result = [raw_vector([1], [1.0])] * count
    with pytest.raises(ValueError, match="one vector per document"):
        sparse.Bm25Encoder(settings()).embed_documents(["one document"])


@pytest.mark.parametrize(
    "output",
    [
        raw_vector([1, 2], [1.0]),
        raw_vector([1, 1], [1.0, 2.0]),
        raw_vector([-1], [1.0]),
        raw_vector([1], [-1.0]),
        raw_vector([1], [float("nan")]),
        raw_vector([1], [float("inf")]),
    ],
)
def test_invalid_backend_vectors_fail_before_storage(
    backend: tuple[FakeModel, list[Settings]], output: SimpleNamespace
) -> None:
    model, _ = backend
    model.query_result = [output]
    with pytest.raises(ValueError):
        sparse.Bm25Encoder(settings()).embed_query("question")


def test_constructor_takes_a_configuration_snapshot(
    backend: tuple[FakeModel, list[Settings]],
) -> None:
    _, loads = backend
    configuration = settings(sparse_language="french")
    encoder = sparse.Bm25Encoder(configuration)
    configuration.sparse_language = "english"
    encoder.embed_query("question")
    assert loads[0].sparse_language == "french"


def test_factory_caches_one_lazy_encoder(
    backend: tuple[FakeModel, list[Settings]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, loads = backend
    monkeypatch.setattr(sparse, "get_settings", settings)
    encoder = sparse.get_sparse_encoder()
    assert encoder is sparse.get_sparse_encoder()
    assert loads == []
    encoder.embed_query("a")
    sparse.get_sparse_encoder().embed_query("b")
    assert len(loads) == 1


def test_loader_passes_french_cpu_and_bm25_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    model = FakeModel()

    def constructor(**kwargs: Any) -> FakeModel:
        calls.append(kwargs)
        return model

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(SparseTextEmbedding=constructor))
    configuration = settings(bm25_avg_len=140, bm25_k=1.4, bm25_b=0.6)
    assert sparse._load_model(configuration) is model
    assert calls == [
        {
            "model_name": "Qdrant/bm25",
            "language": "french",
            "cuda": False,
            "k": 1.4,
            "b": 0.6,
            "avg_len": 140.0,
        }
    ]
