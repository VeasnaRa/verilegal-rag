"""Core Pydantic models: chunks, obligations, citations, findings and reports (PROJECT.md 5)."""

import hashlib
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

Status = Literal["COMPLIANT", "PARTIALLY_COMPLIANT", "NON_COMPLIANT", "NOT_APPLICABLE"]
Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
DocType = Literal["regulation_article", "recital", "policy_section"]
ChunkRole = Literal["parent", "child"]


def text_sha256(text: str) -> str:
    """Return the `sha256:<hex>` hash stored in `Chunk.text_hash`."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Chunk(_StrictModel):
    """A retrievable unit of regulation or policy text with its structural metadata."""

    chunk_id: str = Field(min_length=1)
    doc_type: DocType
    role: ChunkRole = "child"  # children are retrieved; parents (full articles) feed the judge
    text: str = Field(min_length=1)
    source_document: str
    language: str = "fr"
    parent_id: str | None = None
    text_hash: str
    section_path: list[str] = Field(default_factory=list)  # headings above the chunk

    # Regulation-only metadata
    regulation: str | None = None
    celex: str | None = None
    jurisdiction: str | None = None
    chapter: str | None = None
    article_number: PositiveInt | None = None
    article_title: str | None = None
    paragraph: PositiveInt | None = None
    point: str | None = None
    recital_number: PositiveInt | None = None
    source_url: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.text_hash != text_sha256(self.text):
            raise ValueError("text_hash does not match text")
        if self.role == "parent" and self.parent_id is not None:
            raise ValueError("a parent chunk cannot have a parent_id")
        return self


class Obligation(_StrictModel):
    """One checkable requirement from the reviewed obligation catalog (data/catalog/*.yaml)."""

    id: str = Field(min_length=1)
    regulation: str
    article: PositiveInt
    paragraph: PositiveInt | None = None
    point: str | None = None
    title: str
    requirement: str
    evidence_hints: list[str] = Field(default_factory=list)
    applies_to: list[str] = Field(default_factory=list)
    severity: Severity = "MEDIUM"


class Citation(_StrictModel):
    """A quoted passage tied to the chunk it was taken from."""

    source_document: str
    chunk_id: str
    section_or_article: str
    verbatim_quote: str = Field(max_length=600)
    verified: bool = False  # set by the citation verifier, never by the LLM


class AuditFinding(_StrictModel):
    """The judged coverage of one obligation by the audited policy."""

    obligation_id: str
    requirement_topic: str
    status: Status
    severity: Severity
    evidence_found: bool
    explanation: str
    regulatory_reference: Citation
    policy_reference: Citation | None = None  # None when the gap has nothing to cite
    recommended_action: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _compliant_needs_evidence(self) -> Self:
        if self.status == "COMPLIANT" and (
            not self.evidence_found or self.policy_reference is None
        ):
            raise ValueError("a COMPLIANT finding requires evidence and a policy_reference")
        return self


class ComplianceReport(_StrictModel):
    """The full audit output for one policy document."""

    audit_id: str
    audit_title: str
    policy_document: str
    regulations_in_scope: list[str]
    catalog_version: str
    findings: list[AuditFinding]
    overall_risk: Severity  # computed by audit/scoring.py
    unverified_citation_count: NonNegativeInt
