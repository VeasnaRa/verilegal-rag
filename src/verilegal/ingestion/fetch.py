"""Download regulation XHTML from the Publications Office Cellar API and record a manifest."""

import hashlib
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import structlog
from pydantic import BaseModel, Field
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from verilegal import __version__
from verilegal.ingestion.sources import (
    LANGUAGES,
    RegulationSource,
    get_source,
    manifest_path,
    snapshot_path,
)

log = structlog.get_logger(__name__)

CELLAR_URL = "https://publications.europa.eu/resource/celex/{celex}"
USER_AGENT = f"verilegal/{__version__} (regulatory compliance research)"
XHTML_TYPE = "application/xhtml+xml"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """Raised when Cellar does not return a usable XHTML document."""


class SnapshotRecord(BaseModel):
    """Provenance of one downloaded regulation snapshot."""

    key: str
    regulation: str
    celex: str
    language: str
    file: str
    request_url: str
    resolved_url: str
    eurlex_url: str
    content_type: str
    fetched_at: datetime
    size_bytes: int
    sha256: str


class Manifest(BaseModel):
    """All snapshots fetched for one language, keyed by source key."""

    language: str
    snapshots: dict[str, SnapshotRecord] = Field(default_factory=dict)


def load_manifest(raw_dir: Path, lang: str) -> Manifest:
    """Read the manifest for `lang`, or return an empty one if none exists yet."""
    path = manifest_path(raw_dir, lang)
    if not path.exists():
        return Manifest(language=lang)
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(manifest: Manifest, raw_dir: Path) -> Path:
    """Write the manifest as stable, human-readable JSON."""
    path = manifest_path(raw_dir, manifest.language)
    ordered = Manifest(
        language=manifest.language, snapshots=dict(sorted(manifest.snapshots.items()))
    )
    path.write_text(ordered.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in RETRYABLE_STATUS


def _download(
    client: httpx.Client, url: str, headers: dict[str, str], attempts: int, wait: wait_base
) -> httpx.Response:
    retrying = Retrying(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(attempts),
        wait=wait,
        reraise=True,
    )
    for attempt in retrying:
        with attempt:
            response = client.get(url, headers=headers)
            if response.status_code in RETRYABLE_STATUS:
                log.warning("cellar_retryable_status", url=url, status=response.status_code)
                response.raise_for_status()
    return response


def fetch_regulation(
    client: httpx.Client,
    source: RegulationSource,
    lang: str,
    raw_dir: Path,
    *,
    attempts: int = 5,
    wait: wait_base | None = None,
) -> SnapshotRecord:
    """Download one regulation's XHTML into `raw_dir` and return its provenance record."""
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language {lang!r}; expected one of {sorted(LANGUAGES)}")
    url = CELLAR_URL.format(celex=source.celex)
    headers = {"Accept": XHTML_TYPE, "Accept-Language": LANGUAGES[lang]}
    try:
        response = _download(
            client, url, headers, attempts, wait or wait_exponential(multiplier=2, max=60)
        )
    except httpx.HTTPError as exc:
        raise FetchError(
            f"{source.name}: download failed after {attempts} attempts: {exc}"
        ) from exc

    content_type = response.headers.get("content-type", "")
    if response.status_code != 200:
        raise FetchError(f"{source.name}: Cellar answered HTTP {response.status_code} for {url}")
    if not content_type.startswith(XHTML_TYPE) or b"<html" not in response.content[:2000]:
        raise FetchError(f"{source.name}: expected XHTML, got {content_type!r} from {response.url}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(raw_dir, source.key, lang)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(response.content)
    tmp.replace(path)

    record = SnapshotRecord(
        key=source.key,
        regulation=source.name,
        celex=source.celex,
        language=lang,
        file=path.name,
        request_url=url,
        resolved_url=str(response.url),
        eurlex_url=source.eurlex_url(lang),
        content_type=content_type,
        fetched_at=datetime.now(UTC),
        size_bytes=len(response.content),
        sha256=hashlib.sha256(response.content).hexdigest(),
    )
    log.info("cellar_fetched", source=source.key, lang=lang, bytes=record.size_bytes)
    return record


def fetch_regulations(
    keys: Iterable[str],
    lang: str,
    raw_dir: Path,
    *,
    client: httpx.Client | None = None,
    delay_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 5,
    wait: wait_base | None = None,
) -> Manifest:
    """Fetch several regulations politely and merge their records into the manifest."""
    sources = [get_source(key) for key in keys]
    manifest = load_manifest(raw_dir, lang)
    own_client = client is None
    http = client or httpx.Client(
        follow_redirects=True, timeout=60.0, headers={"User-Agent": USER_AGENT}
    )
    try:
        for index, source in enumerate(sources):
            if index:
                sleep(delay_seconds)
            record = fetch_regulation(http, source, lang, raw_dir, attempts=attempts, wait=wait)
            manifest.snapshots[source.key] = record
            save_manifest(manifest, raw_dir)  # persist progress after each download
    finally:
        if own_client:
            http.close()
    return manifest
