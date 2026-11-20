"""Real BGE-M3 demonstration; explicitly run with pytest -s to display actual outputs."""

import numpy as np
import pytest
import torch

from verilegal.config import get_settings
from verilegal.domain.models import Chunk
from verilegal.indexing.embeddings import BgeM3Embedder

pytestmark = pytest.mark.integration


def test_real_dense_embeddings(regulation_examples: list[Chunk], example_question: str) -> None:
    """Encode actual regulation passages and print vectors plus illustrative similarities."""
    settings = get_settings()
    encoder = BgeM3Embedder(settings)
    cuda_requested = encoder.device == "cuda"
    if cuda_requested:
        torch.cuda.reset_peak_memory_stats()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 4))
    try:
        matrix = np.asarray(
            encoder.embed_documents([chunk.text for chunk in regulation_examples]), dtype=np.float32
        )
        query_vector = np.asarray(encoder.embed_query(example_question), dtype=np.float32)
    finally:
        torch.set_num_threads(previous_threads)

    assert matrix.shape == (len(regulation_examples), 1024)
    assert query_vector.shape == (1024,)
    assert np.isfinite(matrix).all() and np.isfinite(query_vector).all()
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(query_vector), 1.0, atol=1e-5)

    print(f"\n=== REAL DENSE ENCODER: {settings.embedding_model_name} ===")
    print(f"Device: {encoder.device}; batch size: {encoder.batch_size}")
    print(f"Document matrix: {matrix.shape}; token limit: {settings.max_seq_length}")
    for chunk, vector in zip(regulation_examples, matrix, strict=True):
        print(f"\n{chunk.chunk_id} | {chunk.regulation} | {chunk.article_title}")
        print("Source excerpt:", " ".join(chunk.text.split())[:200])
        print("First 8 vector values:", np.array2string(vector[:8], precision=6, floatmode="fixed"))
        print(f"Vector norm: {np.linalg.norm(vector):.6f}")
    print("\nQuestion:", example_question)
    print(
        "First 8 query values:", np.array2string(query_vector[:8], precision=6, floatmode="fixed")
    )
    similarities = matrix @ query_vector
    assert np.isfinite(similarities).all()
    print("Cosine similarity on these examples only (not a retrieval benchmark):")
    for index in np.argsort(-similarities):
        print(f"  {float(similarities[index]):.6f}  {regulation_examples[index].chunk_id}")
    if cuda_requested:
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"CUDA peak allocated memory: {peak:.3f} GiB")
        assert peak < 3.5, "Reduce batch/token limits to keep peak allocated VRAM below 3.5 GiB."
