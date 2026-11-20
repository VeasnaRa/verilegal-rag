"""Parent/child legal chunking (article → paragraph → point) and a regex fallback for plain text.

Parents are whole articles (given to the judge); children are paragraphs (retrieved). Oversized
children are split per point — see docs/DECISIONS.md (2026-10-02, chunk sizing).
"""

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from verilegal.domain.models import Chunk, text_sha256
from verilegal.ingestion.regulation_parser import (
    Article,
    Block,
    Paragraph,
    ParsedRegulation,
    clean_text,
)


def _chunk(**fields: Any) -> Chunk:
    return Chunk(text_hash=text_sha256(fields["text"]), **fields)


def split_paragraph(paragraph: Paragraph, max_chars: int) -> list[tuple[str, str | None, str]]:
    """Split a long paragraph based on point such as a), b).

    Each point chunk is prefixed with the paragraph's lead-in sentence so it stays readable on its
    own; text blocks after the first point become `_al{n}` chunks (n = block position).
    """
    text = paragraph.text
    blocks = paragraph.blocks
    if len(text) <= max_chars or len(blocks) < 2:
        return [("", None, text)]

    first_point = next((i for i, block in enumerate(blocks) if block.point), None)
    if first_point is None:
        return [(f"_al{i}", None, block.text) for i, block in enumerate(blocks, start=1)]

    lead = "\n".join(block.text for block in blocks[:first_point])
    seen: Counter[str] = Counter()
    units: list[tuple[str, str | None, str]] = []
    for position, block in enumerate(blocks[first_point:], start=first_point + 1):
        if block.point is None:
            units.append((f"_al{position}", None, block.text))
            continue
        seen[block.point] += 1
        suffix = f"_pt{block.point}" if seen[block.point] == 1 else f"_pt{block.point}-{position}"
        units.append((suffix, block.point, f"{lead}\n{block.text}" if lead else block.text))
    return units


def chunk_regulation(
    parsed: ParsedRegulation,
    *,
    doc_id: str,
    name: str,
    lang: str,
    max_chars: int,
    jurisdiction: str = "EU",
    celex: str | None = None,
    source_url: str | None = None,
) -> list[Chunk]:
    """Turn a parsed regulation into parent articles, child paragraphs and recital chunks."""
    chunks: list[Chunk] = []
    common: dict[str, Any] = {
        "source_document": name,
        "language": lang,
        "regulation": name,
        "celex": celex,
        "jurisdiction": jurisdiction,
    }

    for article in parsed.articles:
        parent_id = f"{doc_id}_art{article.number}"
        article_fields = {
            **common,
            "doc_type": "regulation_article",
            "chapter": article.chapter,
            "article_number": article.number,
            "article_title": article.title or None,
            "section_path": article.section_path,
            "source_url": f"{source_url}#art_{article.number}" if source_url else None,
        }
        chunks.append(
            _chunk(chunk_id=parent_id, role="parent", text=article.text, **article_fields)
        )

        unnumbered = 0
        for paragraph in article.paragraphs:
            if paragraph.number is not None:
                base = f"{parent_id}_p{paragraph.number}"
            else:
                unnumbered += 1
                base = f"{parent_id}_body" + (str(unnumbered) if unnumbered > 1 else "")
            for suffix, point, text in split_paragraph(paragraph, max_chars):
                chunks.append(
                    _chunk(
                        chunk_id=base + suffix,
                        parent_id=parent_id,
                        paragraph=paragraph.number,
                        point=point,
                        text=text,
                        **article_fields,
                    )
                )

    for recital in parsed.recitals:
        chunks.append(
            _chunk(
                chunk_id=f"{doc_id}_rct{recital.number}",
                doc_type="recital",
                recital_number=recital.number,
                text=recital.text,
                source_url=f"{source_url}#rct_{recital.number}" if source_url else None,
                **common,
            )
        )
    return chunks


def validate_chunks(chunks: list[Chunk]) -> list[str]:
    """Check chunk-set invariants; return human-readable problems (empty when valid)."""
    problems: list[str] = []
    counts = Counter(chunk.chunk_id for chunk in chunks)
    problems += [f"duplicate chunk_id {cid} (x{n})" for cid, n in counts.items() if n > 1]
    parents = {chunk.chunk_id for chunk in chunks if chunk.role == "parent"}
    with_children = {chunk.parent_id for chunk in chunks if chunk.parent_id}
    problems += [
        f"{chunk.chunk_id}: unknown parent {chunk.parent_id}"
        for chunk in chunks
        if chunk.parent_id and chunk.parent_id not in parents
    ]
    problems += [f"{pid}: parent without children" for pid in sorted(parents - with_children)]
    return problems


def write_chunks_jsonl(chunks: Iterable[Chunk], path: Path) -> int:
    """Write one chunk per line (UTF-8, LF) and return the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [chunk.model_dump_json(exclude_none=True) for chunk in chunks]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return len(lines)


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    """Load and validate chunks written by `write_chunks_jsonl`."""
    with path.open(encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]


# ---- Regex fallback for plain-text French legal documents (Légifrance, CNIL) ----

# A heading is the marker alone, or marker + separator (hyphen, dashes, colon, dot) + title.
# "Article 6 du RGPD s'applique" is therefore body text, not a heading.
_TITLE = r"\s*(?:[-\u2013\u2014:.]\s*(.*))?$"
ARTICLE_LINE = re.compile(r"^Article\s+(premier|1er|\d+)" + _TITLE, re.IGNORECASE)
CHAPTER_LINE = re.compile(r"^CHAPITRE\s+([IVXLC]+|premier|Ier)" + _TITLE)
SECTION_LINE = re.compile(r"^Section\s+(\d+)" + _TITLE)
PARAGRAPH_LINE = re.compile(r"^(\d+)\.\s")
POINT_LINE = re.compile(r"^([a-z])\)\s|^(\d+)°\s")


def _article_number(token: str) -> int:
    return 1 if token.lower() in {"premier", "1er"} else int(token)


def _heading(label: str, title: str) -> str:
    return f"{label} — {title}" if title else label


def parse_plain_text(text: str) -> ParsedRegulation:
    """Parse plain text into articles using line regexes (chapters, sections, §, points).

    Wrapped lines are joined; a blank line or a structural marker starts a new block. Text before
    the first article (preamble, visas) is ignored.
    """
    articles: list[Article] = []
    chapter: str | None = None
    headings: dict[str, str] = {}  # "chapter" / "section" → heading text
    pending_title: str | None = None  # heading level waiting for a title on the next line
    current: dict[str, Any] | None = None
    blocks: list[Block] = []
    paragraph_number: int | None = None
    paragraphs: list[Paragraph] = []
    block_lines: list[str] = []
    block_point: str | None = None

    def end_block() -> None:
        nonlocal block_lines, block_point
        if block_lines:
            blocks.append(Block(text=clean_text(" ".join(block_lines)), point=block_point))
        block_lines, block_point = [], None

    def end_paragraph() -> None:
        nonlocal blocks
        end_block()
        if blocks:
            paragraphs.append(Paragraph(number=paragraph_number, blocks=blocks))
        blocks = []

    def end_article() -> None:
        nonlocal current, paragraphs
        end_paragraph()
        if current is not None:
            articles.append(Article(paragraphs=paragraphs, **current))
        current, paragraphs = None, []

    for raw_line in text.splitlines():
        line = clean_text(raw_line)
        if not line:
            end_block()
            continue
        if match := CHAPTER_LINE.match(line):
            end_article()
            chapter = match.group(1)
            chapter = "I" if chapter.lower() in {"premier", "ier"} else chapter
            headings = {"chapter": _heading(f"CHAPITRE {chapter}", match.group(2) or "")}
            pending_title = None if match.group(2) else "chapter"
            continue
        if match := SECTION_LINE.match(line):
            end_article()
            headings.pop("section", None)
            headings["section"] = _heading(f"Section {match.group(1)}", match.group(2) or "")
            pending_title = None if match.group(2) else "section"
            continue
        if match := ARTICLE_LINE.match(line):
            end_article()
            pending_title = None
            paragraph_number = None
            current = {
                "number": _article_number(match.group(1)),
                "title": match.group(2) or "",
                "chapter": chapter,
                "section_path": list(headings.values()),
            }
            continue
        if current is None:
            if pending_title:
                headings[pending_title] = _heading(headings[pending_title], line)
                pending_title = None
            continue
        if match := PARAGRAPH_LINE.match(line):
            end_paragraph()
            paragraph_number = int(match.group(1))
            block_lines = [line]
            continue
        if match := POINT_LINE.match(line):
            end_block()
            block_point = match.group(1) or match.group(2)
            block_lines = [line]
            continue
        block_lines.append(line)
    end_article()
    return ParsedRegulation(title="", articles=articles, recitals=[])
