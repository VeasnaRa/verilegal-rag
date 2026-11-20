"""Real CPU BM25 demonstration using the same source passages as the dense example."""

import pytest

from verilegal.config import get_settings
from verilegal.domain.models import Chunk
from verilegal.indexing.sparse import Bm25Encoder

pytestmark = pytest.mark.integration


def test_real_sparse_embeddings(regulation_examples: list[Chunk], example_question: str) -> None:
    """Print real sparse term IDs, document/query weights and overlapping query terms."""
    settings = get_settings()
    encoder = Bm25Encoder(settings)
    vectors = encoder.embed_documents([chunk.text for chunk in regulation_examples])
    question = encoder.embed_query(example_question)
    assert len(vectors) == len(regulation_examples)
    assert question.indices, "The demonstration question must contain searchable terms."

    print(f"\n=== REAL SPARSE ENCODER: {settings.sparse_model_name} ===")
    print(f"Device: cpu; preprocessing language: {settings.sparse_language}")
    print("Question:", example_question)
    print("Query (term ID, weight):", list(zip(question.indices, question.values, strict=True)))
    for chunk, vector in zip(regulation_examples, vectors, strict=True):
        assert vector.indices
        print(f"\n{chunk.chunk_id} | {chunk.regulation} | {chunk.article_title}")
        print("Source excerpt:", " ".join(chunk.text.split())[:200])
        print("Nonzero terms:", len(vector.indices))
        preview = [
            (index, round(value, 6))
            for index, value in zip(vector.indices, vector.values, strict=True)
        ][:8]
        print("First 8 (term ID, weight) pairs:", preview)
        overlap = sorted(set(question.indices) & set(vector.indices))
        print("Shared query term IDs:", overlap)

    assert set(question.indices) & set(vectors[0].indices)
    print("\nThese are sparse representations and term overlaps, not final BM25 scores.")
    print("Qdrant's IDF modifier is required for the planned BM25 ranking.")
