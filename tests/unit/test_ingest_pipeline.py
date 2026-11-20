import hashlib
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from verilegal import cli
from verilegal.config import get_settings
from verilegal.ingestion.chunking import read_chunks_jsonl
from verilegal.ingestion.fetch import Manifest, SnapshotRecord, save_manifest
from verilegal.ingestion.pipeline import (
    IngestionError,
    ingest_regulation,
    processed_path,
    render_sanity_report,
)
from verilegal.ingestion.sources import SOURCES, RegulationSource

MINI = RegulationSource("mini", "Mini", "32099R0001", expected_articles=3, expected_recitals=2)


@pytest.fixture
def raw_dir(tmp_path: Path, mini_regulation_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A raw snapshot directory holding the mini regulation and its manifest entry."""
    monkeypatch.setitem(SOURCES, "mini", MINI)
    raw = tmp_path / "data" / "raw" / "regulations"
    raw.mkdir(parents=True)
    snapshot = raw / "mini.fr.xhtml"
    shutil.copyfile(mini_regulation_path, snapshot)
    content = snapshot.read_bytes()
    record = SnapshotRecord(
        key="mini",
        regulation="Mini",
        celex=MINI.celex,
        language="fr",
        file=snapshot.name,
        request_url="https://example.test/celex",
        resolved_url="https://example.test/doc",
        eurlex_url=MINI.eurlex_url("fr"),
        content_type="application/xhtml+xml",
        fetched_at=datetime(2026, 10, 2, tzinfo=UTC),
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    save_manifest(Manifest(language="fr", snapshots={"mini": record}), raw)
    return raw


def test_ingest_writes_validated_jsonl(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    stats = ingest_regulation("mini", "fr", raw_dir, out, max_chars=60)

    assert stats.ok
    assert (stats.articles, stats.recitals) == (3, 2)
    assert (stats.parent_chunks, stats.recital_chunks) == (3, 2)
    assert stats.split_children > 0
    chunks = read_chunks_jsonl(processed_path(out, "mini", "fr"))
    assert len(chunks) == stats.parent_chunks + stats.child_chunks + stats.recital_chunks
    assert stats.output == processed_path(out, "mini", "fr").as_posix()


def test_structure_mismatch_blocks_output(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        SOURCES, "mini", RegulationSource("mini", "Mini", MINI.celex, 4, expected_recitals=2)
    )
    out = tmp_path / "processed"
    stats = ingest_regulation("mini", "fr", raw_dir, out, max_chars=1800)
    assert not stats.ok
    assert stats.output is None
    assert not processed_path(out, "mini", "fr").exists()
    assert "FAILED" in render_sanity_report([stats], "fr")


def test_tampered_snapshot_is_rejected(raw_dir: Path, tmp_path: Path) -> None:
    with (raw_dir / "mini.fr.xhtml").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(IngestionError, match="sha256"):
        ingest_regulation("mini", "fr", raw_dir, tmp_path, max_chars=1800)


def test_missing_snapshot_is_reported(raw_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="verilegal fetch"):
        ingest_regulation("mini", "en", raw_dir, tmp_path, max_chars=1800)


def test_sanity_report_lists_official_counts(raw_dir: Path, tmp_path: Path) -> None:
    stats = ingest_regulation("mini", "fr", raw_dir, tmp_path, max_chars=1800)
    report = render_sanity_report([stats], "fr")
    assert "| Mini | 3 / 3 | 2 / 2 |" in report
    assert "## Problems" not in report


@pytest.fixture
def cli_env(raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("VERILEGAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VERILEGAL_REPORTS_DIR", str(tmp_path / "reports"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_cli_ingest_regulations(cli_env: Path) -> None:
    result = CliRunner().invoke(cli.app, ["ingest", "regulations", "--source", "mini"])
    assert result.exit_code == 0, result.output
    assert (cli_env / "data" / "processed" / "mini.fr.jsonl").exists()
    assert (cli_env / "reports" / "ingestion_sanity.fr.md").exists()


def test_cli_ingest_unknown_source_fails(cli_env: Path) -> None:
    result = CliRunner().invoke(cli.app, ["ingest", "regulations", "--source", "csrd"])
    assert result.exit_code == 1
    assert "unknown source" in result.output


def test_cli_fetch_passes_sources_and_language(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], str, Path]] = []

    def fake_fetch(keys: list[str], lang: str, raw_dir: Path) -> Manifest:
        calls.append((list(keys), lang, raw_dir))
        return Manifest(language=lang)

    monkeypatch.setattr(cli, "fetch_regulations", fake_fetch)
    result = CliRunner().invoke(cli.app, ["fetch", "-s", "rgpd", "-s", "dora", "--lang", "fr"])
    assert result.exit_code == 0, result.output
    assert calls == [(["rgpd", "dora"], "fr", cli_env / "data" / "raw" / "regulations")]
