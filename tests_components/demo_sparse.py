"""Inspect real sparse embeddings: uv run python tests_components/demo_sparse.py."""

import argparse

from _examples import EXAMPLE_QUESTION, load_regulation_examples, print_passage

from verilegal.config import get_settings
from verilegal.indexing.sparse import Bm25Encoder


def main() -> None:
    """Encode real regulation passages and a question, then print BM25 term weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=EXAMPLE_QUESTION, help="Question to encode.")
    question: str = parser.parse_args().question

    # 1. Use the same real source passages and question as the dense example.
    chunks = load_regulation_examples()
    settings = get_settings()
    encoder = Bm25Encoder(settings)
    print(f"=== Sparse encoding: {settings.sparse_model_name} ===", flush=True)
    print(f"Loaded {len(chunks)} passages from {settings.processed_dir}", flush=True)

    # 2. Document encoding and query encoding use different BM25 weights.
    documents = encoder.embed_documents([chunk.text for chunk in chunks])
    query = encoder.embed_query(question)
    print(f"Device: cpu; preprocessing language: {settings.sparse_language}")
    print("Question:", question)
    print("Query (term ID, weight):", list(zip(query.indices, query.values, strict=True)))

    # 3. Sparse vectors store only nonzero term IDs and their weights.
    for chunk, vector in zip(chunks, documents, strict=True):
        print_passage(chunk)
        print("Nonzero terms:", len(vector.indices))
        preview = [
            (index, round(value, 6))
            for index, value in zip(vector.indices, vector.values, strict=True)
        ][:8]
        print("First 8 (term ID, weight) pairs:", preview)
        print("Shared query term IDs:", sorted(set(query.indices) & set(vector.indices)))

    print("\nThese are sparse vectors and term overlaps; final BM25 ranking needs Qdrant IDF.")


if __name__ == "__main__":
    main()
