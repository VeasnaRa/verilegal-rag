"""CPU BM25 sparse encoding; the eventual Qdrant collection must enable Modifier.IDF."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

import structlog

from verilegal.config import Settings, get_settings
from verilegal.indexing._validation import validate_texts

if TYPE_CHECKING:
    from fastembed.sparse.sparse_embedding_base import SparseEmbedding

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SparseVector:
    """An immutable, sorted sparse vector independent of the Qdrant client."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("sparse indices and values must have matching lengths")
        if any(index < 0 for index in self.indices):
            raise ValueError("sparse indices must be nonnegative")
        if list(self.indices) != sorted(set(self.indices)):
            raise ValueError("sparse indices must be sorted and unique")
        if any(not math.isfinite(value) or value < 0 for value in self.values):
            raise ValueError("BM25 sparse values must be finite and nonnegative")


class _SparseModel(Protocol):
    def embed(self, documents: list[str], *, batch_size: int) -> Iterable[SparseEmbedding]: ...

    def query_embed(self, query: str) -> Iterable[SparseEmbedding]: ...


def _load_model(settings: Settings) -> _SparseModel:
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding(
        model_name=settings.sparse_model_name,
        language=settings.sparse_language,
        cuda=False,
        k=settings.bm25_k,
        b=settings.bm25_b,
        avg_len=settings.bm25_avg_len,
    )
    return cast(_SparseModel, model)


def _to_vector(embedding: SparseEmbedding) -> SparseVector:
    pairs = sorted(
        zip(
            (int(index) for index in embedding.indices),
            (float(value) for value in embedding.values),
            strict=True,
        )
    )
    return SparseVector(
        indices=tuple(index for index, _ in pairs), values=tuple(value for _, value in pairs)
    )


class Bm25Encoder:
    """Lazily encode documents and questions with fixed, CPU-only BM25 preprocessing."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = (settings or get_settings()).model_copy(deep=True)
        self._model: _SparseModel | None = None

    def _get_model(self) -> _SparseModel:
        if self._model is None:
            self._model = _load_model(self._settings)
            log.info(
                "sparse_model_loaded",
                model=self._settings.sparse_model_name,
                language=self._settings.sparse_language,
                device="cpu",
            )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Return BM25 document vectors in input order without dense-style normalization."""
        documents = validate_texts(texts)
        if not documents:
            return []
        vectors = [
            _to_vector(item)
            for item in self._get_model().embed(
                documents, batch_size=self._settings.sparse_batch_size
            )
        ]
        if len(vectors) != len(documents):
            raise ValueError("BM25 must return exactly one vector per document")
        return vectors

    def embed_query(self, text: str) -> SparseVector:
        """Use BM25 query weights, which differ from document term-frequency weights."""
        validate_texts([text])
        vectors = [_to_vector(item) for item in self._get_model().query_embed(text)]
        if len(vectors) != 1:
            raise ValueError("BM25 must return exactly one vector per query")
        return vectors[0]


@lru_cache(maxsize=1)
def get_sparse_encoder() -> Bm25Encoder:
    """Return the process-wide BM25 encoder; preprocessing files load on first use."""
    return Bm25Encoder(get_settings())
