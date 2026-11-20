"""Real regulation passages for explicitly invoked component demonstrations."""

import pytest
from _examples import EXAMPLE_QUESTION, load_regulation_examples

from verilegal.domain.models import Chunk


@pytest.fixture(scope="session")
def regulation_examples() -> list[Chunk]:
    """Load real security, backup and AI-transparency passages from processed resources."""
    try:
        return load_regulation_examples()
    except (FileNotFoundError, ValueError) as error:
        pytest.fail(str(error))


@pytest.fixture(scope="session")
def example_question() -> str:
    """Return a real-time-style question to encode alongside the regulation passages."""
    return EXAMPLE_QUESTION
