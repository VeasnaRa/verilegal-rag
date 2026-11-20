"""Shared input validation for encoders, preserving the original source text."""

from collections.abc import Sequence


def validate_texts(texts: Sequence[str]) -> list[str]:
    """Copy an ordered text batch, rejecting bare strings and blank/non-string items."""
    if isinstance(texts, str):
        raise TypeError("expected a sequence of texts, not a single string; use embed_query")
    result = list(texts)
    for index, text in enumerate(result):
        if not isinstance(text, str):
            raise TypeError(f"text at position {index} must be a string")
        if not text.strip():
            raise ValueError(f"text at position {index} must not be blank")
    return result
