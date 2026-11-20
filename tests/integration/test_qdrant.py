import pytest
from qdrant_client import QdrantClient

from verilegal.config import get_settings

pytestmark = pytest.mark.integration


def test_qdrant_is_reachable() -> None:
    client = QdrantClient(url=get_settings().qdrant_url, timeout=5)
    client.get_collections()
