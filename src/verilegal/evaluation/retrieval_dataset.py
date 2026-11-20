"""Load retrieval judgments and validate their evidence against a fixed regulation corpus."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from verilegal.domain.models import Chunk
from verilegal.ingestion.chunking import read_chunks_jsonl
from verilegal.ingestion.pipeline import IngestionError, ingest_regulation
from verilegal.ingestion.sources import SOURCES

SourceKey = Literal["rgpd", "dora", "aiact"]
QueryCategory = Literal["exact_reference", "fr_paraphrase", "en_query", "cross_regulation"]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class DatasetError(ValueError):
    """Raised when a dataset cannot be read or has duplicate records."""


class RelevanceEvidence(BaseModel):
    """A source excerpt and text hash supporting one positive chunk label."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: NonEmptyText
    text_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    quote: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]


class RetrievalQuery(BaseModel):
    """One retrieval query, its search scope, and explicitly judged relevant chunks."""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyText
    query: NonEmptyText
    language: Literal["fr", "en"]
    category: QueryCategory
    regulations: list[SourceKey] = Field(min_length=1)
    relevant_chunk_ids: list[NonEmptyText] = Field(min_length=1)
    evidence: list[RelevanceEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent_labels(self) -> Self:
        for name in ("regulations", "relevant_chunk_ids"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {name}")
        evidence_ids = [item.chunk_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate evidence chunk_id")
        if set(evidence_ids) != set(self.relevant_chunk_ids):
            raise ValueError("evidence must cover exactly the relevant_chunk_ids")
        if self.category == "cross_regulation":
            if len(self.regulations) < 2:
                raise ValueError("cross_regulation requires at least two regulations")
        elif len(self.regulations) != 1:
            raise ValueError("single-regulation categories require exactly one regulation")
        if self.category == "en_query" and self.language != "en":
            raise ValueError("en_query requires language=en")
        if self.category == "fr_paraphrase" and self.language != "fr":
            raise ValueError("fr_paraphrase requires language=fr")
        return self


def load_retrieval_queries(path: Path) -> list[RetrievalQuery]:
    """Read UTF-8 JSONL, rejecting invalid rows, duplicate IDs and repeated question text."""
    queries: list[RetrievalQuery] = []
    ids: set[str] = set()
    texts: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                query = RetrievalQuery.model_validate_json(line)
            except ValidationError as exc:
                raise DatasetError(f"{path}:{line_number}: {exc}") from exc
            normalized = " ".join(query.query.casefold().split())
            if query.id in ids:
                raise DatasetError(f"{path}:{line_number}: duplicate id {query.id!r}")
            if normalized in texts:
                raise DatasetError(f"{path}:{line_number}: duplicate question {query.query!r}")
            ids.add(query.id)
            texts.add(normalized)
            queries.append(query)
    if not queries:
        raise DatasetError(f"{path}: dataset is empty")
    return queries


def validate_retrieval_queries(
    queries: Sequence[RetrievalQuery], chunks: Iterable[Chunk]
) -> list[str]:
    """Check IDs, retrievable roles, regulation coverage, source hashes and verbatim excerpts."""
    problems: list[str] = []
    corpus: dict[str, Chunk] = {}
    for candidate in chunks:
        if candidate.chunk_id in corpus:
            problems.append(f"duplicate corpus chunk_id {candidate.chunk_id}")
        corpus[candidate.chunk_id] = candidate
    names_to_keys = {source.name: key for key, source in SOURCES.items()}
    for query in queries:
        covered: set[str] = set()
        for evidence in query.evidence:
            prefix = f"{query.id}: {evidence.chunk_id}"
            chunk = corpus.get(evidence.chunk_id)
            if chunk is None:
                problems.append(f"{prefix}: unknown chunk_id")
                continue
            if chunk.role != "child" or chunk.doc_type not in {"regulation_article", "recital"}:
                problems.append(f"{prefix}: label must target a retrievable regulation chunk")
            source_key = names_to_keys.get(chunk.regulation or "")
            if source_key is None or source_key not in query.regulations:
                problems.append(f"{prefix}: outside query regulation scope")
            else:
                covered.add(source_key)
            if evidence.text_hash != chunk.text_hash:
                problems.append(f"{prefix}: source text_hash changed; review the label")
            if evidence.quote not in chunk.text:
                problems.append(f"{prefix}: evidence quote is absent from source text")
        if covered != set(query.regulations):
            problems.append(f"{query.id}: evidence must cover every regulation in scope")
    return problems


def load_retrieval_corpus(
    queries: Sequence[RetrievalQuery], raw_dir: Path, *, max_chars: int
) -> list[Chunk]:
    """Rebuild scoped chunks from verified local snapshots without relying on ignored JSONL."""
    chunks: list[Chunk] = []
    sources = sorted({key for query in queries for key in query.regulations})
    with TemporaryDirectory(prefix="verilegal-benchmark-") as temporary:
        for key in sources:
            stats = ingest_regulation(key, "fr", raw_dir, Path(temporary), max_chars=max_chars)
            if not stats.ok or stats.output is None:
                raise IngestionError("; ".join(stats.problems))
            chunks.extend(read_chunks_jsonl(Path(stats.output)))
    return chunks
