import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from tenacity import wait_none

from verilegal.ingestion.fetch import (
    CELLAR_URL,
    FetchError,
    fetch_regulation,
    fetch_regulations,
    load_manifest,
)
from verilegal.ingestion.sources import SOURCES, manifest_path, snapshot_path

XHTML = b'<?xml version="1.0" encoding="UTF-8"?><html xmlns="http://www.w3.org/1999/xhtml"><body>ok</body></html>'
MANIFESTATION = "http://publications.europa.eu/resource/cellar/abc.0009.03/DOC_1"
XHTML_HEADERS = {"content-type": "application/xhtml+xml;charset=UTF-8"}

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def _cellar_ok(requests: list[httpx.Request]) -> Handler:
    """Mimic Cellar: 303 from the CELEX URI to a manifestation, which serves XHTML."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/resource/celex/"):
            return httpx.Response(303, headers={"location": MANIFESTATION})
        return httpx.Response(200, content=XHTML, headers=XHTML_HEADERS)

    return handler


def test_fetch_follows_redirect_and_writes_snapshot(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    with _client(_cellar_ok(requests)) as client:
        record = fetch_regulation(client, SOURCES["rgpd"], "fr", tmp_path)

    first = requests[0]
    assert str(first.url) == CELLAR_URL.format(celex="32016R0679")
    assert first.headers["accept"] == "application/xhtml+xml"
    assert first.headers["accept-language"] == "fra"

    path = snapshot_path(tmp_path, "rgpd", "fr")
    assert path.read_bytes() == XHTML
    assert record.file == path.name
    assert record.resolved_url == MANIFESTATION
    assert record.sha256 == hashlib.sha256(XHTML).hexdigest()
    assert record.size_bytes == len(XHTML)
    assert record.eurlex_url.endswith("CELEX:32016R0679")
    assert not list(tmp_path.glob("*.part"))


def test_fetch_regulations_writes_and_merges_manifest(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []
    sleeps: list[float] = []
    with _client(_cellar_ok(requests)) as client:
        fetch_regulations(["rgpd"], "fr", tmp_path, client=client, sleep=sleeps.append)
        fetch_regulations(["dora", "aiact"], "fr", tmp_path, client=client, sleep=sleeps.append)

    manifest = load_manifest(tmp_path, "fr")
    assert sorted(manifest.snapshots) == ["aiact", "dora", "rgpd"]
    assert sleeps == [1.0]  # polite delay between consecutive downloads only
    raw = json.loads(manifest_path(tmp_path, "fr").read_text(encoding="utf-8"))
    assert list(raw["snapshots"]) == ["aiact", "dora", "rgpd"]


def test_retries_transient_errors(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=XHTML, headers=XHTML_HEADERS)

    with _client(handler) as client:
        fetch_regulation(client, SOURCES["dora"], "fr", tmp_path, wait=wait_none())
    assert calls["n"] == 3


def test_gives_up_after_max_attempts(tmp_path: Path) -> None:
    with (
        _client(lambda request: httpx.Response(429)) as client,
        pytest.raises(FetchError, match="after 2 attempts"),
    ):
        fetch_regulation(client, SOURCES["dora"], "fr", tmp_path, attempts=2, wait=wait_none())


def test_does_not_retry_client_errors(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    with _client(handler) as client, pytest.raises(FetchError, match="HTTP 404"):
        fetch_regulation(client, SOURCES["rgpd"], "fr", tmp_path, wait=wait_none())
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(300, content=b"<html>choices</html>", headers=XHTML_HEADERS),
        httpx.Response(200, content=b"PK\x03\x04zip", headers={"content-type": "application/zip"}),
        httpx.Response(200, content=b"not html", headers=XHTML_HEADERS),
    ],
)
def test_rejects_non_xhtml_responses(tmp_path: Path, response: httpx.Response) -> None:
    with _client(lambda request: response) as client, pytest.raises(FetchError):
        fetch_regulation(client, SOURCES["rgpd"], "fr", tmp_path)
    assert not snapshot_path(tmp_path, "rgpd", "fr").exists()


def test_rejects_unknown_source_and_language(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown source"):
        fetch_regulations(["csrd"], "fr", tmp_path)
    with _client(_cellar_ok([])) as client, pytest.raises(ValueError, match="language"):
        fetch_regulation(client, SOURCES["rgpd"], "de", tmp_path)
