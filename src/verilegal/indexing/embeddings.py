"""Lazy BGE-M3 dense encoding with normalized output, batching and CUDA OOM back-off."""

import gc
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, Self, cast

import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from verilegal.config import Device, Settings, get_settings, resolve_device
from verilegal.indexing._validation import validate_texts

log = structlog.get_logger(__name__)
DenseVector = list[float]


class Embedder(Protocol):
    """The dense encoder interface shared by future indexing and retrieval pipelines."""

    @property
    def dimension(self) -> int:
        """Return the size of each output vector."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[DenseVector]:
        """Encode a batch of documents while preserving their order."""
        ...

    def embed_query(self, text: str) -> DenseVector:
        """Encode one question in the same vector space as documents."""
        ...


class _DenseModel(Protocol):
    """Small backend interface so unit tests can substitute a model without downloads."""

    max_seq_length: int

    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> NDArray[np.float32]: ...

    def to(self, device: str) -> Self: ...

    def float(self) -> Self: ...


def _load_model(model_name: str, device: Device) -> _DenseModel:
    from sentence_transformers import SentenceTransformer

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": dtype})
    return cast(_DenseModel, model)


class BgeM3Embedder:
    """Generate dense text embeddings using a BGE-M3 model loaded on first use per instance."""

    dimension = 1024

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        if settings.embedder != "bge-m3":
            raise NotImplementedError("mistral-embed is not implemented; select embedder=bge-m3")
        self._model_name = settings.embedding_model_name
        self._max_seq_length = settings.max_seq_length
        self._batch_size = settings.embed_batch_size
        self._device = resolve_device(settings.device)
        self._model: _DenseModel | None = None

    @property
    def device(self) -> Device:
        """Return the current device, including any CPU fallback after a CUDA OOM."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Return the current batch size, including reductions made after a CUDA OOM."""
        return self._batch_size

    def _get_model(self) -> _DenseModel:
        if self._model is not None:
            return self._model
        try:
            self._model = _load_model(self._model_name, self._device)
        except torch.cuda.OutOfMemoryError:
            if self._device != "cuda":
                raise
            # A smaller text batch cannot fix an OOM while loading model weights.
            self._device = "cpu"
            log.warning("dense_load_oom_cpu_fallback", model=self._model_name)
        if self._model is None:
            gc.collect()
            torch.cuda.empty_cache()
            self._model = _load_model(self._model_name, "cpu")
        if self._model.get_sentence_embedding_dimension() != self.dimension:
            self._model = None
            raise ValueError("the BGE-M3 encoder requires a 1024-dimensional model")
        self._model.max_seq_length = self._max_seq_length
        log.info(
            "dense_model_loaded",
            model=self._model_name,
            device=self._device,
            gpu_name=torch.cuda.get_device_name(0) if self._device == "cuda" else None,
        )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[DenseVector]:
        """Return finite, normalized dense vectors in input order; empty batches return []."""
        documents = validate_texts(texts)
        if not documents:
            return []
        model = self._get_model()
        while True:
            try:
                vectors = model.encode(
                    documents,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if self._device != "cuda":
                    raise
                log.warning("dense_encode_oom", batch_size=self._batch_size)
            # Leave the exception handler before clearing cached memory, releasing its traceback.
            if self._batch_size > 1:
                self._batch_size = max(1, self._batch_size // 2)
            else:
                model = model.to("cpu").float()
                self._model = model
                self._device = "cpu"
                log.warning("dense_encode_oom_cpu_fallback", model=self._model_name)
            gc.collect()
            torch.cuda.empty_cache()
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(documents), self.dimension):
            raise ValueError(f"unexpected dense embedding shape: {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("dense embeddings must contain finite values")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError("dense embeddings must have finite, nonzero norms")
        # Normalize in float32 as well, avoiding fp16 rounding in stored vectors.
        return cast(list[DenseVector], (matrix / norms).tolist())

    def embed_query(self, text: str) -> DenseVector:
        """Encode one question with the same BGE-M3 model and normalization as documents."""
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the process-wide encoder; weights load on the first nonempty encoding call."""
    return BgeM3Embedder(get_settings())
