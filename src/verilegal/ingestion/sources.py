"""Registry of the regulations VeriLegal can fetch and ingest (PROJECT.md Appendix A)."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegulationSource:
    """A regulation identified by its CELEX number, with its official structure counts."""

    key: str
    name: str
    celex: str
    expected_articles: int
    expected_recitals: int

    def eurlex_url(self, lang: str) -> str:
        """Human-readable EUR-Lex link used in citations."""
        return f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/?uri=CELEX:{self.celex}"


SOURCES: dict[str, RegulationSource] = {
    "rgpd": RegulationSource(
        "rgpd", "RGPD", "32016R0679", expected_articles=99, expected_recitals=173
    ),
    "dora": RegulationSource(
        "dora", "DORA", "32022R2554", expected_articles=64, expected_recitals=106
    ),
    "aiact": RegulationSource(
        "aiact", "AI Act", "32024R1689", expected_articles=113, expected_recitals=180
    ),
}

# ISO 639-1 (used in file names and settings) → ISO 639-2/B (expected by Cellar's Accept-Language)
LANGUAGES: dict[str, str] = {"fr": "fra", "en": "eng"}


def get_source(key: str) -> RegulationSource:
    """Return the registered source for `key`, with a helpful error for unknown keys."""
    try:
        return SOURCES[key]
    except KeyError:
        raise ValueError(f"unknown source {key!r}; expected one of {sorted(SOURCES)}") from None


def snapshot_path(raw_dir: Path, key: str, lang: str) -> Path:
    """Location of a fetched XHTML snapshot."""
    return raw_dir / f"{key}.{lang}.xhtml"


def manifest_path(raw_dir: Path, lang: str) -> Path:
    """Location of the snapshot manifest for one language."""
    return raw_dir / f"manifest.{lang}.json"
