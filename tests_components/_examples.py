"""Load real regulation passages for scripts and optional integration tests."""

from pathlib import Path

from verilegal.config import get_settings
from verilegal.domain.models import Chunk
from verilegal.ingestion.chunking import read_chunks_jsonl

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    ("rgpd", "32016R0679_art32_p1"),
    ("dora", "32022R2554_art12_p1"),
    ("aiact", "32024R1689_art50_p1"),
)
EXAMPLE_QUESTION = (
    "Quelles mesures de sécurité protègent les données personnelles par le chiffrement ?"
)


def load_regulation_examples() -> list[Chunk]:
    """Read three real child chunks from the configured processed regulation files."""
    directory = get_settings().processed_dir
    if not directory.is_absolute():
        directory = ROOT / directory
    examples: list[Chunk] = []
    for source, chunk_id in EXAMPLES:
        path = directory / f"{source}.fr.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run `uv run verilegal ingest regulations` first."
            )
        chunks = {chunk.chunk_id: chunk for chunk in read_chunks_jsonl(path)}
        if chunk_id not in chunks:
            raise ValueError(
                f"Missing example {chunk_id} in {path}; check the chunking configuration."
            )
        chunk = chunks[chunk_id]
        if chunk.role != "child" or chunk.doc_type != "regulation_article":
            raise ValueError(f"Example {chunk_id} must be a child regulation article chunk.")
        examples.append(chunk)
    return examples


def print_passage(chunk: Chunk) -> None:
    """Print the source identity and full text of a sampled regulation chunk."""
    print(f"\n{chunk.chunk_id} | {chunk.regulation} | {chunk.article_title}")
    print("Source:", chunk.source_document)
    print("Text:", chunk.text)
