from pathlib import Path

import pytest

from verilegal.ingestion.regulation_parser import (
    ParsedRegulation,
    check_structure,
    clean_text,
    parse_regulation_file,
)
from verilegal.ingestion.sources import RegulationSource


@pytest.fixture
def parsed(mini_regulation_path: Path) -> ParsedRegulation:
    return parse_regulation_file(mini_regulation_path)


def test_title_and_counts(parsed: ParsedRegulation) -> None:
    assert parsed.title == "RÈGLEMENT (UE) 2099/1 DU PARLEMENT EUROPÉEN relatif aux essais"
    assert [a.number for a in parsed.articles] == [1, 2, 3]
    assert [r.number for r in parsed.recitals] == [1, 2]


def test_article_premier_uses_id_number_and_title(parsed: ParsedRegulation) -> None:
    article = parsed.articles[0]
    assert article.number == 1
    assert article.title == "Objet"


def test_numbered_paragraph_boundaries(parsed: ParsedRegulation) -> None:
    paragraphs = parsed.articles[0].paragraphs
    assert [p.number for p in paragraphs] == [1, 2]
    assert paragraphs[0].text == "1. Le présent règlement établit des règles."
    assert paragraphs[1].text.startswith("2. Le responsable du traitement met en œuvre")
    assert paragraphs[1].text.endswith("sans préjudice de l'article 2.")


def test_points_with_nested_subpoints(parsed: ParsedRegulation) -> None:
    blocks = parsed.articles[0].paragraphs[1].blocks
    assert [b.point for b in blocks] == [None, "a", "b", None]
    assert blocks[1].text == "a) la pseudonymisation et le chiffrement des données;"
    assert blocks[2].text == "b) des moyens permettant de garantir:\ni) la confidentialité;"


def test_unnumbered_article_body(parsed: ParsedRegulation) -> None:
    (body,) = parsed.articles[1].paragraphs
    assert body.number is None
    assert [b.point for b in body.blocks] == [None, "1", "2"]
    assert body.blocks[0].text == "Aux fins du présent règlement, on entend par:"


def test_quoted_amendment_is_not_a_paragraph(parsed: ParsedRegulation) -> None:
    (body,) = parsed.articles[2].paragraphs
    assert body.number is None
    assert body.text == (
        "À l’article 17 du règlement (UE) 2018/1, le paragraphe suivant est ajouté:\n"
        "«3. Les États membres veillent à l’application.»"
    )


def test_chapter_and_section_headings(parsed: ParsedRegulation) -> None:
    first, _, third = parsed.articles
    assert first.chapter == "I"
    assert first.section_path == ["CHAPITRE I — Dispositions générales"]
    assert third.chapter == "II"
    assert third.section_path == [
        "CHAPITRE II — Obligations",
        "Section 1 — Obligations générales",
    ]


def test_recitals_strip_label_and_footnotes(parsed: ParsedRegulation) -> None:
    first, second = parsed.recitals
    assert first.text == "La protection des données est un droit fondamental."
    assert second.text == "Le règlement no 45/2001 du Conseil, s'applique."


def test_parsing_is_deterministic(mini_regulation_path: Path) -> None:
    assert parse_regulation_file(mini_regulation_path) == parse_regulation_file(
        mini_regulation_path
    )


def test_clean_text_normalizes_spaces() -> None:
    assert clean_text("  1.\xa0\xa0 Texte\n  suivi ,  fin .") == "1. Texte suivi, fin."


def test_check_structure_against_expected_counts(parsed: ParsedRegulation) -> None:
    good = RegulationSource("mini", "Mini", "32099R0001", expected_articles=3, expected_recitals=2)
    assert check_structure(parsed, good) == []

    wrong = RegulationSource("mini", "Mini", "32099R0001", expected_articles=4, expected_recitals=1)
    problems = check_structure(parsed, wrong)
    assert any("expected articles 1..4" in p for p in problems)
    assert any("expected recitals 1..1" in p for p in problems)
