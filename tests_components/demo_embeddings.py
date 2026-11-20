"""Inspect real dense embeddings: uv run python tests_components/demo_embeddings.py."""

import argparse

import numpy as np
import torch
from _examples import EXAMPLE_QUESTION, load_regulation_examples, print_passage

from verilegal.config import get_settings
from verilegal.indexing.embeddings import BgeM3Embedder


def main() -> None:
    """Generate and display dense embeddings for regulation passages and a question."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=EXAMPLE_QUESTION, help="Question to encode.")
    question: str = parser.parse_args().question

    # 1. Read actual chunks produced by the ingestion pipeline.
    chunks = load_regulation_examples()
    settings = get_settings()
    encoder = BgeM3Embedder(settings)
    print(f"=== Dense encoding: {settings.embedding_model_name} ===", flush=True)
    print(f"Loaded {len(chunks)} passages from {settings.processed_dir}", flush=True)

    # 2. Call our implementation; it loads the real model on the first encoding call.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    documents = np.asarray(
        encoder.embed_documents([chunk.text for chunk in chunks]), dtype=np.float32
    )
    query = np.asarray(encoder.embed_query(question), dtype=np.float32)

    # 3. Inspect each passage and its vector. Print a preview to keep the output readable.
    print(f"Device: {encoder.device}; document matrix shape: {documents.shape}")
    for chunk, vector in zip(chunks, documents, strict=True):
        print_passage(chunk)
        print("Vector shape:", vector.shape)
        print("First 8 values:", np.array2string(vector[:8], precision=6, floatmode="fixed"))
        print(f"Vector norm: {np.linalg.norm(vector):.6f}")

    print("\nQuestion:", question)
    print("Query vector shape:", query.shape)
    print("First 8 query values:", np.array2string(query[:8], precision=6, floatmode="fixed"))

    # 4. Normalized vectors let us compute cosine similarity with a dot product.
    similarities = documents @ query
    print("\nCosine similarity against these three passages (not a retrieval benchmark):")
    for index in np.argsort(-similarities):
        print(f"  {float(similarities[index]):.6f}  {chunks[index].chunk_id}")


if __name__ == "__main__":
    main()
