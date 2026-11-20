from pathlib import Path

import pytest

from verilegal.domain.models import Chunk, text_sha256
from verilegal.ingestion.chunking import (
    chunk_regulation,
    parse_plain_text,
    read_chunks_jsonl,
    split_paragraph,
    validate_chunks,
    write_chunks_jsonl,
)
from verilegal.ingestion.regulation_parser import Block, Paragraph, parse_regulation_file

CELEX = "32099R0001"
URL = f"https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:{CELEX}"


def _chunks(path: Path, max_chars: int = 10_000) -> dict[str, Chunk]:
    chunks = chunk_regulation(
        parse_regulation_file(path),
        doc_id=CELEX,
        name="Mini",
        lang="fr",
        max_chars=max_chars,
        celex=CELEX,
        source_url=URL,
    )
    return {chunk.chunk_id: chunk for chunk in chunks}


def test_parent_child_and_recital_ids(mini_regulation_path: Path) -> None:
    chunks = _chunks(mini_regulation_path)
    assert list(chunks) == [
        f"{CELEX}_art1",
        f"{CELEX}_art1_p1",
        f"{CELEX}_art1_p2",
        f"{CELEX}_art2",
        f"{CELEX}_art2_body",
        f"{CELEX}_art3",
        f"{CELEX}_art3_body",
        f"{CELEX}_rct1",
        f"{CELEX}_rct2",
    ]


def test_metadata(mini_regulation_path: Path) -> None:
    chunks = _chunks(mini_regulation_path)
    parent = chunks[f"{CELEX}_art1"]
    child = chunks[f"{CELEX}_art1_p2"]
    recital = chunks[f"{CELEX}_rct2"]

    assert parent.role == "parent"
    assert parent.parent_id is None
    assert parent.text == f"{chunks[f'{CELEX}_art1_p1'].text}\n\n{child.text}"

    assert child.role == "child"
    assert child.parent_id == parent.chunk_id
    assert child.doc_type == "regulation_article"
    assert (child.regulation, child.celex, child.jurisdiction, child.language) == (
        "Mini",
        CELEX,
        "EU",
        "fr",
    )
    assert (child.chapter, child.article_number, child.article_title, child.paragraph) == (
        "I",
        1,
        "Objet",
        2,
    )
    assert child.source_url == f"{URL}#art_1"
    assert child.text_hash == text_sha256(child.text)

    assert recital.doc_type == "recital"
    assert recital.recital_number == 2
    assert recital.parent_id is None
    assert recital.source_url == f"{URL}#rct_2"


def test_ids_and_hashes_are_stable_across_runs(mini_regulation_path: Path) -> None:
    first = _chunks(mini_regulation_path, max_chars=60)
    second = _chunks(mini_regulation_path, max_chars=60)
    assert {k: c.text_hash for k, c in first.items()} == {k: c.text_hash for k, c in second.items()}


def test_oversized_paragraphs_split_per_point(mini_regulation_path: Path) -> None:
    chunks = _chunks(mini_regulation_path, max_chars=60)
    art1_children = [cid for cid in chunks if cid.startswith(f"{CELEX}_art1_p2")]
    assert art1_children == [f"{CELEX}_art1_p2_pta", f"{CELEX}_art1_p2_ptb", f"{CELEX}_art1_p2_al4"]

    point_b = chunks[f"{CELEX}_art1_p2_ptb"]
    assert point_b.point == "b"
    assert point_b.paragraph == 2
    assert point_b.text == (
        "2. Le responsable du traitement met en œuvre, selon les besoins:\n"
        "b) des moyens permettant de garantir:\ni) la confidentialité;"
    )
    trailing = chunks[f"{CELEX}_art1_p2_al4"]
    assert trailing.point is None
    assert trailing.text == "Le présent paragraphe s'applique sans préjudice de l'article 2."

    definitions = [cid for cid in chunks if cid.startswith(f"{CELEX}_art2_body")]
    assert definitions == [f"{CELEX}_art2_body_pt1", f"{CELEX}_art2_body_pt2"]
    assert chunks[f"{CELEX}_art2_body_pt2"].text.startswith("Aux fins du présent règlement")
    assert validate_chunks(list(chunks.values())) == []


def test_split_without_points_and_duplicate_labels() -> None:
    no_points = Paragraph(number=1, blocks=[Block(text="x" * 50), Block(text="y" * 50)])
    assert [s for s, _, _ in split_paragraph(no_points, max_chars=60)] == ["_al1", "_al2"]

    single_block = Paragraph(number=1, blocks=[Block(text="z" * 500)])
    assert split_paragraph(single_block, max_chars=60) == [("", None, "z" * 500)]

    duplicated = Paragraph(
        number=1,
        blocks=[
            Block(text="Intro:"),
            Block(text="a) " + "x" * 40, point="a"),
            Block(text="a) " + "y" * 40, point="a"),
        ],
    )
    assert [s for s, _, _ in split_paragraph(duplicated, max_chars=60)] == ["_pta", "_pta-3"]


def test_validate_chunks_reports_problems() -> None:
    def chunk(cid: str, role: str = "child", parent: str | None = None) -> Chunk:
        return Chunk(
            chunk_id=cid,
            doc_type="regulation_article",
            role=role,  # type: ignore[arg-type]
            text="t",
            text_hash=text_sha256("t"),
            source_document="X",
            parent_id=parent,
        )

    problems = validate_chunks(
        [chunk("a", "parent"), chunk("b", parent="missing"), chunk("b"), chunk("c", "parent")]
    )
    assert "duplicate chunk_id b (x2)" in problems
    assert "b: unknown parent missing" in problems
    assert "a: parent without children" in problems


def test_jsonl_round_trip(tmp_path: Path, mini_regulation_path: Path) -> None:
    chunks = list(_chunks(mini_regulation_path).values())
    path = tmp_path / "out" / "mini.fr.jsonl"
    assert write_chunks_jsonl(chunks, path) == len(chunks)
    assert read_chunks_jsonl(path) == chunks
    assert path.read_bytes().count(b"\n") == len(chunks)
    assert "œuvre" in path.read_text(encoding="utf-8")  # UTF-8, not \u escapes


PLAIN_LAW = """\
Loi relative à l'informatique, aux fichiers et aux libertés

CHAPITRE Ier
Principes et définitions

Article premier
L'informatique doit être au service de chaque citoyen.
Son développement doit s'opérer dans le cadre de la coopération internationale.

Article 2 - Champ d'application
1. La présente loi s'applique aux traitements automatisés.
2. Elle s'applique également aux traitements suivants :
a) les traitements mis en œuvre par un responsable établi en France ;
b) les traitements portant sur des données
relatives à des personnes résidant en France.
Le décret précise les modalités.

CHAPITRE II : Conditions de licéité
Section 1
Dispositions générales

Article 3
Un traitement ne peut porter que sur des données adéquates :
1° Collectées pour des finalités déterminées ;
2° Exactes et complètes.
Article 6 du RGPD s'applique sans préjudice des présentes dispositions.
"""


def test_plain_text_fallback_structure() -> None:
    parsed = parse_plain_text(PLAIN_LAW)
    assert [a.number for a in parsed.articles] == [1, 2, 3]
    first, second, third = parsed.articles

    assert first.title == ""
    assert first.chapter == "I"
    assert first.section_path == ["CHAPITRE I — Principes et définitions"]
    (body,) = first.paragraphs
    assert body.number is None
    assert body.text == (
        "L'informatique doit être au service de chaque citoyen. "
        "Son développement doit s'opérer dans le cadre de la coopération internationale."
    )

    assert second.title == "Champ d'application"
    assert [p.number for p in second.paragraphs] == [1, 2]
    blocks = second.paragraphs[1].blocks
    assert [b.point for b in blocks] == [None, "a", "b"]
    assert blocks[2].text == (
        "b) les traitements portant sur des données relatives à des personnes résidant en France. "
        "Le décret précise les modalités."
    )

    assert third.chapter == "II"
    assert third.section_path == [
        "CHAPITRE II — Conditions de licéité",
        "Section 1 — Dispositions générales",
    ]
    points = [b.point for b in third.paragraphs[0].blocks]
    assert points == [None, "1", "2"]
    # "Article 6 du RGPD …" is body text, not a new article
    assert "Article 6 du RGPD" in third.paragraphs[0].blocks[-1].text


def test_plain_text_fallback_chunks() -> None:
    chunks = chunk_regulation(
        parse_plain_text(PLAIN_LAW),
        doc_id="LOI-78-17",
        name="Loi Informatique et Libertés",
        lang="fr",
        max_chars=10_000,
        jurisdiction="FR",
    )
    ids = [c.chunk_id for c in chunks]
    assert ids[:5] == [
        "LOI-78-17_art1",
        "LOI-78-17_art1_body",
        "LOI-78-17_art2",
        "LOI-78-17_art2_p1",
        "LOI-78-17_art2_p2",
    ]
    assert all(c.jurisdiction == "FR" and c.celex is None for c in chunks)
    assert validate_chunks(chunks) == []


@pytest.mark.parametrize("heading", ["Article 1er", "ARTICLE PREMIER", "Article premier - Objet"])
def test_plain_text_article_one_variants(heading: str) -> None:
    parsed = parse_plain_text(f"{heading}\nTexte.")
    assert [a.number for a in parsed.articles] == [1]
