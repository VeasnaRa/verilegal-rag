"""Shared pytest configuration: fixtures directory and `gpu` auto-skip when CUDA is absent."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_regulation_path() -> Path:
    """A small Cellar-style XHTML regulation mirroring the real markup (DECISIONS 2026-10-02)."""
    return FIXTURES / "regulation_mini.xhtml"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked `gpu` on machines without a CUDA device (CI, Docker)."""
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if not gpu_items:
        return

    import torch

    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device")
    for item in gpu_items:
        item.add_marker(skip_gpu)
