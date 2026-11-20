"""Application settings loaded from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DeviceSetting = Literal["auto", "cuda", "cpu"]
Device = Literal["cuda", "cpu"]


class Settings(BaseSettings):
    """Runtime configuration. Variables use the `VERILEGAL_` prefix except third-party secrets."""

    model_config = SettingsConfigDict(
        env_prefix="VERILEGAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    env: Literal["dev", "test", "prod"] = "dev"
    qdrant_url: str = "http://localhost:6333"
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")

    # Ingestion
    max_chunk_chars: PositiveInt = 1800  # ≈ 450 BGE-M3 tokens; longer units split per point

    # Compute
    device: DeviceSetting = "auto"
    embedder: Literal["bge-m3", "mistral-embed"] = "bge-m3"
    embedding_model_name: str = Field(default="BAAI/bge-m3", min_length=1)
    embed_batch_size: PositiveInt = 16
    rerank_batch_size: PositiveInt = 8
    max_seq_length: PositiveInt = 512
    docling_device: Device = "cpu"
    reranker_enabled: bool = True
    retrieval_top_k: PositiveInt = 20
    rerank_top_n: PositiveInt = 5

    # Sparse retrieval (CPU)
    sparse_model_name: str = Field(default="Qdrant/bm25", min_length=1)
    sparse_language: Literal["french", "english"] = "french"
    sparse_batch_size: PositiveInt = 64
    bm25_k: float = Field(default=1.2, gt=0, allow_inf_nan=False)
    bm25_b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    bm25_avg_len: float = Field(default=256.0, gt=0, allow_inf_nan=False)

    # LLM
    llm_provider: Literal["mistral"] = "mistral"
    llm_judge_model: str = "mistral-large-latest"
    llm_fast_model: str = "mistral-small-latest"
    llm_max_concurrency: PositiveInt = 1
    llm_max_retries: NonNegativeInt = 6
    mistral_api_key: SecretStr | None = Field(default=None, validation_alias="MISTRAL_API_KEY")

    # Observability
    langfuse_enabled: bool = Field(default=False, validation_alias="LANGFUSE_ENABLED")
    langfuse_public_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    @property
    def regulations_raw_dir(self) -> Path:
        """Committed Cellar snapshots and their manifests."""
        return self.data_dir / "raw" / "regulations"

    @property
    def processed_dir(self) -> Path:
        """Generated chunk JSONL files (git-ignored, reproducible)."""
        return self.data_dir / "processed"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()


def resolve_device(setting: DeviceSetting) -> Device:
    """Map a device setting to a concrete device: `auto` picks CUDA when available, else CPU."""
    if setting == "cpu":
        return "cpu"

    import torch  # deferred: importing torch is slow and not needed for CPU-only paths

    cuda_available = torch.cuda.is_available()
    if setting == "cuda" and not cuda_available:
        raise RuntimeError("VERILEGAL_DEVICE=cuda but no CUDA device is available")
    return "cuda" if cuda_available else "cpu"
