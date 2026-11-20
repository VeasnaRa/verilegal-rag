import pytest
import torch
from pydantic import ValidationError

from verilegal.config import Settings, get_settings, resolve_device


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("VERILEGAL_DEVICE", "VERILEGAL_EMBED_BATCH_SIZE", "MISTRAL_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.usefixtures("clean_env")
def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.device == "auto"
    assert settings.docling_device == "cpu"
    assert settings.llm_max_concurrency == 1
    assert settings.mistral_api_key is None


@pytest.mark.usefixtures("clean_env")
def test_reads_prefixed_and_unprefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERILEGAL_DEVICE", "cpu")
    monkeypatch.setenv("VERILEGAL_EMBED_BATCH_SIZE", "4")
    monkeypatch.setenv("MISTRAL_API_KEY", "secret-key")
    settings = Settings(_env_file=None)
    assert settings.device == "cpu"
    assert settings.embed_batch_size == 4
    assert settings.mistral_api_key is not None
    assert settings.mistral_api_key.get_secret_value() == "secret-key"
    assert "secret-key" not in repr(settings)


@pytest.mark.usefixtures("clean_env")
def test_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERILEGAL_DEVICE", "tpu")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_resolve_device_cpu_never_needs_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu") == "cpu"


@pytest.mark.parametrize(("available", "expected"), [(True, "cuda"), (False, "cpu")])
def test_resolve_device_auto(
    monkeypatch: pytest.MonkeyPatch, available: bool, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert resolve_device("auto") == expected


def test_resolve_device_cuda_requested_but_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        resolve_device("cuda")


@pytest.mark.gpu
def test_resolve_device_auto_on_real_gpu() -> None:
    assert resolve_device("auto") == "cuda"


def test_encoder_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.embedding_model_name == "BAAI/bge-m3"
    assert settings.sparse_model_name == "Qdrant/bm25"
    assert settings.sparse_language == "french"
    assert settings.sparse_batch_size == 64
    assert settings.bm25_avg_len == 256.0


def test_encoder_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERILEGAL_EMBEDDING_MODEL_NAME", "local/bge-m3")
    monkeypatch.setenv("VERILEGAL_SPARSE_LANGUAGE", "english")
    monkeypatch.setenv("VERILEGAL_SPARSE_BATCH_SIZE", "4")
    monkeypatch.setenv("VERILEGAL_BM25_AVG_LEN", "150")
    settings = Settings(_env_file=None)
    assert settings.embedding_model_name == "local/bge-m3"
    assert settings.sparse_language == "english"
    assert settings.sparse_batch_size == 4
    assert settings.bm25_avg_len == 150.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_model_name": ""},
        {"sparse_language": "unsupported"},
        {"sparse_batch_size": 0},
        {"bm25_k": 0},
        {"bm25_b": -0.1},
        {"bm25_b": 1.1},
        {"bm25_avg_len": 0},
        {"bm25_avg_len": float("inf")},
    ],
)
def test_invalid_encoder_settings_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)  # type: ignore[arg-type]
