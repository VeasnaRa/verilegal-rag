"""Typer command-line interface: the `verilegal` command."""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from verilegal import __version__
from verilegal.config import get_settings
from verilegal.evaluation.retrieval_dataset import (
    load_retrieval_corpus,
    load_retrieval_queries,
    validate_retrieval_queries,
)
from verilegal.ingestion.fetch import FetchError, fetch_regulations
from verilegal.ingestion.pipeline import IngestionError, ingest_regulation, render_sanity_report
from verilegal.ingestion.sources import SOURCES

app = typer.Typer(
    name="verilegal",
    help=(
        "Obligation-driven compliance auditor for RGPD, DORA and the EU AI Act. "
        "Decision support for compliance teams, not legal advice."
    ),
    no_args_is_help=True,
)
ingest_app = typer.Typer(help="Parse and chunk documents into data/processed/*.jsonl.")
app.add_typer(ingest_app, name="ingest")
eval_app = typer.Typer(help="Validate evaluation datasets; retrieval metrics come later.")
app.add_typer(eval_app, name="eval")
console = Console()

SourceOption = Annotated[
    list[str] | None,
    typer.Option(
        "--source",
        "-s",
        help=f"Regulation to process (repeatable): {', '.join(SOURCES)}. Default: all.",
    ),
]
LangOption = Annotated[str, typer.Option("--lang", help="Language of the text (fr, en).")]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"verilegal {__version__}")
        raise typer.Exit()


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code=1)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Global options."""


@app.command()
def fetch(source: SourceOption = None, lang: LangOption = "fr") -> None:
    """Download regulation XHTML snapshots from the Cellar API and update the manifest."""
    raw_dir = get_settings().regulations_raw_dir
    try:
        manifest = fetch_regulations(source or list(SOURCES), lang, raw_dir)
    except (FetchError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    table = Table("source", "CELEX", "size", "sha256", title=f"Snapshots in {raw_dir}")
    for record in manifest.snapshots.values():
        table.add_row(
            record.key, record.celex, f"{record.size_bytes / 1024:.0f} KB", record.sha256[:12]
        )
    console.print(table)


@ingest_app.callback()
def ingest() -> None:
    """Parse and chunk documents into data/processed/*.jsonl."""


@ingest_app.command("regulations")
def ingest_regulations(source: SourceOption = None, lang: LangOption = "fr") -> None:
    """Parse fetched snapshots, chunk them, validate, and write JSONL + a sanity report."""
    settings = get_settings()
    try:
        stats = [
            ingest_regulation(
                key,
                lang,
                settings.regulations_raw_dir,
                settings.processed_dir,
                max_chars=settings.max_chunk_chars,
            )
            for key in source or list(SOURCES)
        ]
    except (IngestionError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    report = settings.reports_dir / f"ingestion_sanity.{lang}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_sanity_report(stats, lang), encoding="utf-8")

    table = Table(
        "regulation", "articles", "recitals", "parents", "children", "split", "oversized", "status"
    )
    for s in stats:
        table.add_row(
            s.regulation,
            f"{s.articles}/{s.expected_articles}",
            f"{s.recitals}/{s.expected_recitals}",
            str(s.parent_chunks),
            str(s.child_chunks),
            str(s.split_children),
            str(s.oversized_chunks),
            "[green]OK[/green]" if s.ok else "[red]FAILED[/red]",
        )
    console.print(table)
    for s in stats:
        for problem in s.problems:
            console.print(f"[red]problem:[/red] {problem}")
        if s.output:
            console.print(f"wrote {s.output}")
    console.print(f"sanity report: {report.as_posix()}")
    if not all(s.ok for s in stats):
        raise typer.Exit(code=1)


@eval_app.command("validate-retrieval")
def validate_retrieval_dataset(
    dataset: Annotated[
        Path,
        typer.Option("--dataset", exists=True, dir_okay=False, help="Retrieval benchmark JSONL."),
    ] = Path("eval/retrieval_queries.jsonl"),
) -> None:
    """Validate query labels against local snapshots without Qdrant, models or API calls."""
    settings = get_settings()
    try:
        queries = load_retrieval_queries(dataset)
        chunks = load_retrieval_corpus(
            queries, settings.regulations_raw_dir, max_chars=settings.max_chunk_chars
        )
        problems = validate_retrieval_queries(queries, chunks)
    except (OSError, ValueError, IngestionError) as exc:
        raise _fail(str(exc)) from exc
    if problems:
        for problem in problems:
            console.print(f"[red]problem:[/red] {problem}")
        raise typer.Exit(code=1)
    table = Table("category", "queries", title=f"Validated dataset: {dataset}")
    for category, count in sorted(Counter(query.category for query in queries).items()):
        table.add_row(category, str(count))
    console.print(table)
    console.print(
        f"[green]OK[/green]: {len(queries)} queries; all labels, hashes and quotes match."
    )
    console.print("Structural validation passed; human relevance review and retrieval runs remain.")
