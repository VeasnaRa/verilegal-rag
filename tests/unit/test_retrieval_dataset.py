import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from verilegal import cli
from verilegal.domain.models import Chunk, text_sha256
from verilegal.evaluation.retrieval_dataset import (
    DatasetError,
    RetrievalQuery,
    load_retrieval_corpus,
    load_retrieval_queries,
    validate_retrieval_queries,
)

ROOT = Path(__file__).resolve().parents[2]
TEXT = "Les données doivent être limitées à ce qui est nécessaire."


def _chunk(**overrides: Any) -> Chunk:
    fields = {
        "chunk_id": "32016R0679_art5_p1_ptc",
        "doc_type": "regulation_article",
        "text": TEXT,
        "text_hash": text_sha256(TEXT),
        "source_document": "RGPD",
        "regulation": "RGPD",
        "parent_id": "32016R0679_art5",
    }
    return Chunk.model_validate(fields | overrides)


def _row(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "rgpd-example",
        "query": "Quelles données faut-il collecter ?",
        "language": "fr",
        "category": "fr_paraphrase",
        "regulations": ["rgpd"],
        "relevant_chunk_ids": [_chunk().chunk_id],
        "evidence": [
            {
                "chunk_id": _chunk().chunk_id,
                "text_hash": text_sha256(TEXT),
                "quote": "limitées à ce qui est nécessaire",
            }
        ],
    } | overrides


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_committed_benchmark_matches_fresh_snapshot_chunks() -> None:
    queries = load_retrieval_queries(ROOT / "eval" / "retrieval_queries.jsonl")
    corpus = load_retrieval_corpus(queries, ROOT / "data" / "raw" / "regulations", max_chars=1800)
    assert len(queries) == 90
    assert Counter(q.category for q in queries) == {
        "exact_reference": 27,
        "fr_paraphrase": 36,
        "en_query": 18,
        "cross_regulation": 9,
    }
    assert Counter(q.language for q in queries) == {"fr": 72, "en": 18}
    assert validate_retrieval_queries(queries, corpus) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"unexpected": True},
        {"query": "   "},
        {"regulations": []},
        {"regulations": ["rgpd", "rgpd"]},
        {"regulations": ["csrd"]},
        {"regulations": ["rgpd", "dora"]},
        {"relevant_chunk_ids": []},
        {"relevant_chunk_ids": ["x", "x"]},
        {"relevant_chunk_ids": ["missing"]},
        {"evidence": []},
        {"category": "cross_regulation"},
        {"category": "en_query", "language": "fr"},
        {"category": "fr_paraphrase", "language": "en"},
    ],
)
def test_schema_rejects_inconsistent_query_labels(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery.model_validate(_row(**overrides))


def test_evidence_requires_distinct_ids_valid_hashes_and_nonempty_quotes() -> None:
    row = _row()
    with pytest.raises(ValidationError, match="duplicate evidence"):
        RetrievalQuery.model_validate(row | {"evidence": row["evidence"] * 2})
    for overrides in ({"text_hash": "bad"}, {"quote": " "}, {"quote": "x" * 601}):
        evidence = row["evidence"][0] | overrides
        with pytest.raises(ValidationError):
            RetrievalQuery.model_validate(row | {"evidence": [evidence]})


def test_load_valid_utf8_rows(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    _write(path, [_row()])
    (query,) = load_retrieval_queries(path)
    assert query.query == "Quelles données faut-il collecter ?"
    assert validate_retrieval_queries([query], [_chunk()]) == []


@pytest.mark.parametrize(
    ("second", "message"),
    [
        (_row(query="Une autre question ?"), "duplicate id"),
        (_row(id="another", query="QUELLES  DONNÉES faut-il collecter ?"), "duplicate question"),
    ],
)
def test_load_rejects_duplicates(tmp_path: Path, second: dict[str, Any], message: str) -> None:
    path = tmp_path / "queries.jsonl"
    _write(path, [_row(), second])
    with pytest.raises(DatasetError, match=message):
        load_retrieval_queries(path)


@pytest.mark.parametrize("content", ["", " \n", "not json\n", '{"query": "only"}\n'])
def test_load_reports_empty_or_invalid_dataset(tmp_path: Path, content: str) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=r"empty|queries\.jsonl:1"):
        load_retrieval_queries(path)


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ([], "unknown chunk_id"),
        ([_chunk(role="parent", parent_id=None)], "retrievable regulation chunk"),
        ([_chunk(doc_type="policy_section")], "retrievable regulation chunk"),
        ([_chunk(regulation="DORA")], "outside query regulation scope"),
        (
            [_chunk(text="Un texte changé.", text_hash=text_sha256("Un texte changé."))],
            "source text_hash changed",
        ),
        ([_chunk(), _chunk()], "duplicate corpus chunk_id"),
    ],
)
def test_validator_rejects_broken_source_labels(chunks: list[Chunk], message: str) -> None:
    problems = validate_retrieval_queries([RetrievalQuery.model_validate(_row())], chunks)
    assert any(message in problem for problem in problems)


def test_validator_rejects_invented_quote_even_with_matching_hash() -> None:
    row = _row()
    row["evidence"][0]["quote"] = "Une phrase qui ne figure pas dans la source."
    problems = validate_retrieval_queries([RetrievalQuery.model_validate(row)], [_chunk()])
    assert any("quote is absent" in problem for problem in problems)


def test_cross_query_requires_evidence_from_each_act() -> None:
    query = RetrievalQuery.model_validate(
        _row(category="cross_regulation", regulations=["rgpd", "dora"])
    )
    assert any(
        "every regulation in scope" in problem
        for problem in validate_retrieval_queries([query], [_chunk()])
    )
    second = _chunk(chunk_id="32022R2554_art9_p2", regulation="DORA")
    row = query.model_dump()
    row["relevant_chunk_ids"].append(second.chunk_id)
    row["evidence"].append(
        {"chunk_id": second.chunk_id, "text_hash": second.text_hash, "quote": TEXT}
    )
    assert (
        validate_retrieval_queries([RetrievalQuery.model_validate(row)], [_chunk(), second]) == []
    )


def test_cli_validates_dataset_and_fails_on_invalid_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_retrieval_corpus", lambda *args, **kwargs: [_chunk()])
    path = tmp_path / "queries.jsonl"
    _write(path, [_row()])
    result = CliRunner().invoke(cli.app, ["eval", "validate-retrieval", "--dataset", str(path)])
    assert result.exit_code == 0, result.output
    assert "1 queries" in result.output
    assert "human relevance review" in result.output

    row = _row()
    row["evidence"][0]["quote"] = "invented"
    _write(path, [row])
    result = CliRunner().invoke(cli.app, ["eval", "validate-retrieval", "--dataset", str(path)])
    assert result.exit_code == 1
    assert "quote is absent" in result.output


def test_cli_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text("invalid\n", encoding="utf-8")
    result = CliRunner().invoke(cli.app, ["eval", "validate-retrieval", "--dataset", str(path)])
    assert result.exit_code == 1
    assert "queries.jsonl:1" in result.output
