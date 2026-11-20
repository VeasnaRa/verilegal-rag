from typing import Any

import pytest
from pydantic import ValidationError

from verilegal.domain.models import (
    AuditFinding,
    Chunk,
    Citation,
    ComplianceReport,
    Obligation,
    text_sha256,
)

ARTICLE_TEXT = "Le responsable du traitement et le sous-traitant mettent en œuvre des mesures."


def _chunk(**overrides: Any) -> Chunk:
    data: dict[str, Any] = {
        "chunk_id": "32016R0679_art32_p1",
        "doc_type": "regulation_article",
        "text": ARTICLE_TEXT,
        "source_document": "RGPD",
        "parent_id": "32016R0679_art32",
        "text_hash": text_sha256(ARTICLE_TEXT),
        "regulation": "RGPD",
        "celex": "32016R0679",
        "jurisdiction": "EU",
        "chapter": "IV",
        "article_number": 32,
        "article_title": "Sécurité du traitement",
        "paragraph": 1,
    }
    data.update(overrides)
    return Chunk(**data)


def _citation(**overrides: Any) -> Citation:
    data: dict[str, Any] = {
        "source_document": "RGPD",
        "chunk_id": "32016R0679_art32_p1",
        "section_or_article": "Article 32(1)",
        "verbatim_quote": "mettent en œuvre des mesures",
    }
    data.update(overrides)
    return Citation(**data)


def _finding(**overrides: Any) -> AuditFinding:
    data: dict[str, Any] = {
        "obligation_id": "RGPD-32-1-a",
        "requirement_topic": "Pseudonymisation et chiffrement",
        "status": "NON_COMPLIANT",
        "severity": "HIGH",
        "evidence_found": False,
        "explanation": "Aucune mention du chiffrement.",
        "regulatory_reference": _citation(),
        "confidence": 0.8,
    }
    data.update(overrides)
    return AuditFinding(**data)


def test_text_sha256_is_stable_and_prefixed() -> None:
    assert text_sha256("é") == text_sha256("é")
    assert text_sha256("é").startswith("sha256:")
    assert text_sha256("a") != text_sha256("b")


def test_chunk_valid() -> None:
    chunk = _chunk()
    assert chunk.article_number == 32
    assert chunk.language == "fr"


def test_chunk_rejects_wrong_hash() -> None:
    with pytest.raises(ValidationError, match="text_hash"):
        _chunk(text_hash=text_sha256("autre texte"))


def test_chunk_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _chunk(unexpected="x")


def test_obligation_defaults() -> None:
    obligation = Obligation(
        id="RGPD-32-1-a",
        regulation="RGPD",
        article=32,
        paragraph=1,
        point="a",
        title="Pseudonymisation et chiffrement",
        requirement="Mesures appropriées, y compris le chiffrement.",
        evidence_hints=["chiffrement", "AES"],
        applies_to=["controller", "processor"],
    )
    assert obligation.severity == "MEDIUM"


def test_citation_unverified_by_default_and_quote_length_capped() -> None:
    assert _citation().verified is False
    with pytest.raises(ValidationError):
        _citation(verbatim_quote="x" * 601)


def test_gap_finding_without_policy_reference() -> None:
    finding = _finding()
    assert finding.policy_reference is None


def test_compliant_finding_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="COMPLIANT"):
        _finding(status="COMPLIANT", evidence_found=True)
    finding = _finding(
        status="COMPLIANT",
        evidence_found=True,
        policy_reference=_citation(source_document="PSSI", chunk_id="pssi_s4"),
    )
    assert finding.status == "COMPLIANT"


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_finding_confidence_bounds(confidence: float) -> None:
    with pytest.raises(ValidationError):
        _finding(confidence=confidence)


def test_finding_rejects_invalid_status() -> None:
    with pytest.raises(ValidationError):
        _finding(status="MAYBE")


def test_report_round_trips_through_json() -> None:
    report = ComplianceReport(
        audit_id="a1",
        audit_title="Audit PSSI",
        policy_document="pssi.pdf",
        regulations_in_scope=["RGPD"],
        catalog_version="0.1.0",
        findings=[_finding()],
        overall_risk="HIGH",
        unverified_citation_count=0,
    )
    assert ComplianceReport.model_validate_json(report.model_dump_json()) == report


def test_report_rejects_negative_unverified_count() -> None:
    with pytest.raises(ValidationError):
        ComplianceReport(
            audit_id="a1",
            audit_title="Audit",
            policy_document="pssi.pdf",
            regulations_in_scope=["RGPD"],
            catalog_version="0.1.0",
            findings=[],
            overall_risk="LOW",
            unverified_citation_count=-1,
        )
