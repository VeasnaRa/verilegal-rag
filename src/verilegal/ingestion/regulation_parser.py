"""Parse Official Journal XHTML (Cellar "ELI" markup) into articles, paragraphs, points, recitals.

The markup conventions this relies on are documented in docs/DECISIONS.md (2026-10-02).
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel

from verilegal.ingestion.sources import RegulationSource

ARTICLE_ID = re.compile(r"^art_(\d+)$")
PARAGRAPH_ID = re.compile(r"^\d{3}\.(\d{3})$")
RECITAL_ID = re.compile(r"^rct_(\d+)$")
CONTAINER_ID = re.compile(r"^cpt_[^.]+(?:\.sct_[^.]+)*$")  # chapter, or section within one
POINT_LABEL = re.compile(r"^«?\(?([0-9]{1,3}|[a-z]{1,5})[).]$")
SPACE_BEFORE_PUNCT = re.compile(r" (?=[,.])")


class Block(BaseModel):
    """One text block of a paragraph: a point (`a)`, `1)`, `ii)`) or plain text."""

    text: str
    point: str | None = None  # normalized label ("a", "1", "ii") when the block is a point


class Paragraph(BaseModel):
    """A numbered paragraph, or the body of an article without numbered paragraphs."""

    number: int | None
    blocks: list[Block]

    @property
    def text(self) -> str:
        """Full paragraph text, one block per line."""
        return "\n".join(block.text for block in self.blocks)


class Article(BaseModel):
    """An article with its headings and paragraphs."""

    number: int
    title: str
    chapter: str | None
    section_path: list[str]
    paragraphs: list[Paragraph]

    @property
    def text(self) -> str:
        """Full article body, paragraphs separated by blank lines."""
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)


class Recital(BaseModel):
    """A numbered recital (considérant)."""

    number: int
    text: str


class ParsedRegulation(BaseModel):
    """The structured content of one regulation."""

    title: str
    articles: list[Article]
    recitals: list[Recital]


def _classes(tag: Tag) -> list[str]:
    value = tag.get("class")
    if value is None:
        return []
    return value.split() if isinstance(value, str) else list(value)


def _id(tag: Tag) -> str:
    value = tag.get("id")
    return value if isinstance(value, str) else ""


def _children(tag: Tag) -> list[Tag]:
    return [child for child in tag.children if isinstance(child, Tag)]


def clean_text(text: str) -> str:
    """Normalize whitespace (incl. non-breaking spaces) to single spaces."""
    text = " ".join(text.replace("\xa0", " ").split())
    return SPACE_BEFORE_PUNCT.sub("", text)


def _rows(table: Tag) -> list[Tag]:
    body = table.find("tbody", recursive=False)
    container = body if isinstance(body, Tag) else table
    return [row for row in _children(container) if row.name == "tr"]


def _render(tag: Tag) -> str:
    """Render a block-level element as text; table rows become `label text` lines."""
    if tag.name == "table":
        return "\n".join(
            " ".join(filter(None, (_render(cell) for cell in _children(row) if cell.name == "td")))
            for row in _rows(table=tag)
        )
    if tag.name in {"div", "td"}:
        blocks = [child for child in _children(tag) if child.name not in {"col", "colgroup"}]
        if not blocks:
            return clean_text(tag.get_text(""))
        return "\n".join(filter(None, (_render(child) for child in blocks)))
    return clean_text(tag.get_text(""))


def _point_label(table: Tag) -> str | None:
    rows = _rows(table)
    if len(rows) != 1:
        return None
    cells = [cell for cell in _children(rows[0]) if cell.name == "td"]
    if len(cells) != 2:
        return None
    match = POINT_LABEL.match(clean_text(cells[0].get_text("")))
    return match.group(1) if match else None


def _blocks(elements: list[Tag]) -> list[Block]:
    blocks: list[Block] = []
    for element in elements:
        text = _render(element)
        if not text:
            continue
        label = _point_label(element) if element.name == "table" else None
        blocks.append(Block(text=text, point=label))
    return blocks


def _container_heading(container: Tag) -> str:
    label = title = ""
    for child in _children(container):
        if not label and "oj-ti-section-1" in _classes(child):
            label = clean_text(child.get_text(""))
        elif not title and (_id(child).endswith(".tit_1") or "oj-ti-section-2" in _classes(child)):
            title = clean_text(child.get_text(""))
    return " — ".join(filter(None, (label, title)))


def _headings(article: Tag) -> tuple[str | None, list[str]]:
    containers = [
        parent
        for parent in article.parents
        if isinstance(parent, Tag) and parent.name == "div" and CONTAINER_ID.match(_id(parent))
    ]
    containers.reverse()  # outermost (chapter) first
    chapter = _id(containers[0]).removeprefix("cpt_") if containers else None
    return chapter, [heading for c in containers if (heading := _container_heading(c))]


def _parse_article(tag: Tag, number: int) -> Article:
    title = ""
    paragraphs: list[Paragraph] = []
    loose: list[Tag] = []

    def flush_loose() -> None:
        if blocks := _blocks(loose):
            paragraphs.append(Paragraph(number=None, blocks=blocks))
        loose.clear()

    for child in _children(tag):
        if "oj-ti-art" in _classes(child):
            continue
        if "eli-title" in _classes(child):
            title = clean_text(child.get_text(""))
        elif match := PARAGRAPH_ID.match(_id(child)):
            flush_loose()
            paragraphs.append(
                Paragraph(number=int(match.group(1)), blocks=_blocks(_children(child)))
            )
        else:
            loose.append(child)
    flush_loose()

    chapter, section_path = _headings(tag)
    return Article(
        number=number,
        title=title,
        chapter=chapter,
        section_path=section_path,
        paragraphs=paragraphs,
    )


def _parse_recital(tag: Tag, number: int) -> Recital:
    table = tag.find("table")
    if isinstance(table, Tag) and (rows := _rows(table)):
        cells = [cell for cell in _children(rows[0]) if cell.name == "td"]
        if len(cells) == 2:
            return Recital(number=number, text=_render(cells[1]))
    return Recital(number=number, text=_render(tag))


def parse_regulation_xhtml(content: bytes | str) -> ParsedRegulation:
    """Parse a Cellar XHTML document into articles and recitals, in document order."""
    soup = BeautifulSoup(content, "lxml-xml")

    # Footnote references, e.g. `<a>(<span class="oj-super oj-note-tag">1</span>)</a>`
    for span in soup.find_all("span"):
        if "oj-note-tag" in _classes(span) and not span.decomposed:
            anchor = span.find_parent("a")
            (anchor if isinstance(anchor, Tag) else span).decompose()

    title_div = soup.find(lambda t: t.name == "div" and "eli-main-title" in _classes(t))
    title = clean_text(title_div.get_text(" ")) if isinstance(title_div, Tag) else ""

    articles: list[Article] = []
    recitals: list[Recital] = []
    for div in soup.find_all("div"):
        if match := ARTICLE_ID.match(_id(div)):
            articles.append(_parse_article(div, int(match.group(1))))
        elif match := RECITAL_ID.match(_id(div)):
            recitals.append(_parse_recital(div, int(match.group(1))))
    return ParsedRegulation(title=title, articles=articles, recitals=recitals)


def parse_regulation_file(path: Path) -> ParsedRegulation:
    """Parse a snapshot file (bytes are passed through so the XML declaration sets the encoding)."""
    return parse_regulation_xhtml(path.read_bytes())


def check_structure(parsed: ParsedRegulation, source: RegulationSource) -> list[str]:
    """Compare parsed structure with the official counts; return human-readable problems."""
    problems: list[str] = []
    article_numbers = [article.number for article in parsed.articles]
    if article_numbers != list(range(1, source.expected_articles + 1)):
        problems.append(
            f"{source.name}: expected articles 1..{source.expected_articles}, "
            f"parsed {len(article_numbers)} ({article_numbers[:3]}…{article_numbers[-3:]})"
        )
    recital_numbers = [recital.number for recital in parsed.recitals]
    if recital_numbers != list(range(1, source.expected_recitals + 1)):
        problems.append(
            f"{source.name}: expected recitals 1..{source.expected_recitals}, "
            f"parsed {len(recital_numbers)}"
        )
    for article in parsed.articles:
        if not article.title:
            problems.append(f"{source.name}: article {article.number} has no title")
        if not article.text.strip():
            problems.append(f"{source.name}: article {article.number} has no text")
        numbers = [p.number for p in article.paragraphs if p.number is not None]
        if numbers and numbers != list(range(1, len(numbers) + 1)):
            problems.append(f"{source.name}: article {article.number} paragraphs {numbers}")
    for recital in parsed.recitals:
        if not recital.text.strip():
            problems.append(f"{source.name}: recital {recital.number} has no text")
    return problems
