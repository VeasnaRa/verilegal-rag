import sys
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from verilegal.config import Settings
from verilegal.indexing import embeddings


class FakeModel:
    def __init__(self) -> None:
        self.max_seq_length = 8192
        self.dimension: int | None = 1024
        self.calls: list[dict[str, Any]] = []
        self.failures: list[Exception] = []
        self.result: NDArray[np.float32] | None = None
        self.moves: list[str] = []
        self.float_calls = 0

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.dimension

    def encode(self, sentences: list[str], **kwargs: Any) -> NDArray[np.float32]:
        self.calls.append({"sentences": list(sentences), **kwargs})
        if self.failures:
            raise self.failures.pop(0)
        if self.result is not None:
            return self.result
        matrix = np.zeros((len(sentences), 1024), dtype=np.float32)
        for row, text in enumerate(sentences):
            column = sum(text.encode("utf-8")) % 1023
            matrix[row, column : column + 2] = [3.0, 4.0]
        return matrix

    def to(self, device: str) -> "FakeModel":
        self.moves.append(device)
        return self

    def float(self) -> "FakeModel":
        self.float_calls += 1
        return self


def settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **({"device": "cpu"} | overrides))


@pytest.fixture(autouse=True)
def isolate_encoder(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    embeddings.get_embedder.cache_clear()
    monkeypatch.setattr(embeddings, "resolve_device", lambda value: value)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Fake GPU")
    yield
    embeddings.get_embedder.cache_clear()


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeModel, list[tuple[str, str]]]:
    model = FakeModel()
    loads: list[tuple[str, str]] = []

    def load(name: str, device: str) -> FakeModel:
        loads.append((name, device))
        return model

    monkeypatch.setattr(embeddings, "_load_model", load)
    return model, loads


def test_dense_vectors_preserve_order_and_are_normalized(
    backend: tuple[FakeModel, list[tuple[str, str]]],
) -> None:
    model, loads = backend
    encoder = embeddings.BgeM3Embedder(
        settings(embedding_model_name="local/bge-m3", embed_batch_size=4, max_seq_length=256)
    )
    assert loads == []
    texts = ["Chiffrement des données", "Résilience opérationnelle"]
    vectors = encoder.embed_documents(texts)
    assert np.asarray(vectors).shape == (2, 1024)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0], atol=1e-6)
    assert vectors[0] != vectors[1]
    assert vectors[0] == encoder.embed_query(texts[0])
    assert vectors[::-1] == encoder.embed_documents(texts[::-1])
    assert loads == [("local/bge-m3", "cpu")]
    assert model.max_seq_length == 256
    assert model.calls[0] == {
        "sentences": texts,
        "batch_size": 4,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }


def test_empty_batch_does_not_load_weights(
    backend: tuple[FakeModel, list[tuple[str, str]]],
) -> None:
    _, loads = backend
    assert embeddings.BgeM3Embedder(settings()).embed_documents([]) == []
    assert loads == []


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_blank_queries_fail_before_model_loading(
    backend: tuple[FakeModel, list[tuple[str, str]]], text: str
) -> None:
    _, loads = backend
    with pytest.raises(ValueError, match="blank"):
        embeddings.BgeM3Embedder(settings()).embed_query(text)
    assert loads == []


@pytest.mark.parametrize("texts", ["not a batch", ["valid", 42], ["valid", None]])
def test_invalid_batch_types_fail_without_loading(
    backend: tuple[FakeModel, list[tuple[str, str]]], texts: Any
) -> None:
    _, loads = backend
    with pytest.raises(TypeError):
        embeddings.BgeM3Embedder(settings()).embed_documents(texts)
    assert loads == []


def test_batch_rejects_blank_item_before_partial_encoding(
    backend: tuple[FakeModel, list[tuple[str, str]]],
) -> None:
    model, _ = backend
    with pytest.raises(ValueError, match="position 1"):
        embeddings.BgeM3Embedder(settings()).embed_documents(["valid", " "])
    assert model.calls == []


def test_cuda_oom_halves_batch_and_reuses_successful_size(
    backend: tuple[FakeModel, list[tuple[str, str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, loads = backend
    model.failures = [torch.cuda.OutOfMemoryError("batch"), torch.cuda.OutOfMemoryError("batch")]
    cleared: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))
    encoder = embeddings.BgeM3Embedder(settings(device="cuda", embed_batch_size=8))
    vectors = encoder.embed_documents(["a", "b"])
    assert [call["batch_size"] for call in model.calls] == [8, 4, 2]
    assert all(call["sentences"] == ["a", "b"] for call in model.calls)
    assert len(vectors) == 2
    assert encoder.device == "cuda"
    assert encoder.batch_size == 2
    assert len(cleared) == 2
    assert loads == [("BAAI/bge-m3", "cuda")]
    encoder.embed_query("a")
    assert model.calls[-1]["batch_size"] == 2


def test_cuda_oom_at_batch_one_moves_model_to_float32_cpu(
    backend: tuple[FakeModel, list[tuple[str, str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, loads = backend
    model.failures = [torch.cuda.OutOfMemoryError("batch") for _ in range(3)]
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    encoder = embeddings.BgeM3Embedder(settings(device="cuda", embed_batch_size=4))
    assert len(encoder.embed_query("question")) == 1024
    assert [call["batch_size"] for call in model.calls] == [4, 2, 1, 1]
    assert encoder.device == "cpu"
    assert model.moves == ["cpu"]
    assert model.float_calls == 1
    assert len(loads) == 1


def test_cuda_weight_loading_oom_falls_back_without_reloading_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    model = FakeModel()

    def load(name: str, device: str) -> FakeModel:
        calls.append(device)
        if device == "cuda":
            raise torch.cuda.OutOfMemoryError("weights")
        return model

    monkeypatch.setattr(embeddings, "_load_model", load)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    encoder = embeddings.BgeM3Embedder(settings(device="cuda"))
    encoder.embed_query("question")
    encoder.embed_query("another question")
    assert calls == ["cuda", "cpu"]
    assert encoder.device == "cpu"


@pytest.mark.parametrize("error", [RuntimeError("backend failure"), torch.cuda.OutOfMemoryError()])
def test_cpu_errors_propagate_without_retrying(
    backend: tuple[FakeModel, list[tuple[str, str]]], error: Exception
) -> None:
    model, _ = backend
    model.failures = [error]
    with pytest.raises(type(error)):
        embeddings.BgeM3Embedder(settings()).embed_query("question")
    assert len(model.calls) == 1


@pytest.mark.parametrize("dimension", [None, 384])
def test_rejects_incompatible_model_dimension(
    backend: tuple[FakeModel, list[tuple[str, str]]], dimension: int | None
) -> None:
    model, _ = backend
    model.dimension = dimension
    with pytest.raises(ValueError, match="1024-dimensional"):
        embeddings.BgeM3Embedder(settings()).embed_query("question")


@pytest.mark.parametrize("shape", [(1, 384), (2, 1024), (1024,)])
def test_rejects_incorrect_output_geometry(
    backend: tuple[FakeModel, list[tuple[str, str]]], shape: tuple[int, ...]
) -> None:
    model, _ = backend
    model.result = np.ones(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        embeddings.BgeM3Embedder(settings()).embed_query("question")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0])
def test_rejects_nonfinite_or_zero_vectors(
    backend: tuple[FakeModel, list[tuple[str, str]]], value: float
) -> None:
    model, _ = backend
    model.result = np.full((1, 1024), value, dtype=np.float32)
    with pytest.raises(ValueError, match=r"finite|nonzero"):
        embeddings.BgeM3Embedder(settings()).embed_query("question")


def test_default_factory_returns_one_lazy_encoder(
    backend: tuple[FakeModel, list[tuple[str, str]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, loads = backend
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings())
    first = embeddings.get_embedder()
    assert first is embeddings.get_embedder()
    assert first.dimension == 1024
    assert loads == []
    first.embed_query("a")
    embeddings.get_embedder().embed_query("b")
    assert len(loads) == 1


def test_unimplemented_provider_fails_explicitly() -> None:
    with pytest.raises(NotImplementedError, match="mistral-embed"):
        embeddings.BgeM3Embedder(settings(embedder="mistral-embed"))


@pytest.mark.parametrize(("device", "dtype"), [("cpu", torch.float32), ("cuda", torch.float16)])
def test_loader_passes_device_and_dtype_without_loading_real_models(
    monkeypatch: pytest.MonkeyPatch, device: str, dtype: torch.dtype
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    fake = FakeModel()

    def constructor(name: str, **kwargs: Any) -> FakeModel:
        calls.append((name, kwargs))
        return fake

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor)
    )
    loader: Callable[..., Any] = embeddings._load_model
    assert loader("configured/model", device) is fake
    assert calls == [
        ("configured/model", {"device": device, "model_kwargs": {"torch_dtype": dtype}})
    ]
