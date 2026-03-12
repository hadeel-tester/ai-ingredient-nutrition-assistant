"""Tests for knowledge_base/build_kb.py — pure functions only.

Covers: parse_frontmatter (YAML parsing + normalisation),
chunk_by_section (section splitting + metadata attachment),
and load_md_files (file discovery + integration of parse + chunk).

build_chroma() is excluded — it calls OpenAI embeddings and ChromaDB,
which are integration-level concerns and not suited for unit testing.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from knowledge_base.build_kb import parse_frontmatter, chunk_by_section, load_md_files

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

SAMPLE_MD = """\
---
ingredient: BHA_BHT
category: antioxidant
e_number: E320_E321
aliases: ["BHA", "BHT", "E320", "E321"]
risk_level: moderate
eu_status: permitted
allergen: false
vegan: true
---

## Summary
BHA and BHT are synthetic antioxidants.

## Health Risks
- High doses cause tumors in animal studies.
"""

SAMPLE_MD_NULL_FIELD = """\
---
ingredient: test_ingredient
category: flavouring
e_number: null
aliases: []
risk_level: low
eu_status: permitted
allergen: false
vegan: true
---

## Summary
A test ingredient with a null e_number.
"""

SAMPLE_MD_NO_FRONTMATTER = """\
# Just a plain heading

Some text without any YAML frontmatter block.
"""


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_happy_path_returns_metadata_and_body(self):
        """Valid YAML frontmatter must be parsed into a dict; body starts after ---."""
        metadata, body = parse_frontmatter(SAMPLE_MD)

        assert metadata["ingredient"] == "BHA_BHT"
        assert metadata["category"] == "antioxidant"
        assert metadata["risk_level"] == "moderate"
        assert metadata["eu_status"] == "permitted"
        assert metadata["allergen"] is False
        assert metadata["vegan"] is True
        assert "## Summary" in body

    def test_list_field_serialised_to_comma_string(self):
        """aliases list must be joined into a comma-separated string for ChromaDB."""
        metadata, _ = parse_frontmatter(SAMPLE_MD)
        assert metadata["aliases"] == "BHA, BHT, E320, E321"

    def test_none_field_normalised_to_empty_string(self):
        """A null YAML value must be stored as an empty string, not Python None."""
        metadata, _ = parse_frontmatter(SAMPLE_MD_NULL_FIELD)
        assert metadata["e_number"] == ""

    def test_no_frontmatter_raises_value_error(self):
        """A markdown file without --- delimiters must raise ValueError."""
        with pytest.raises(ValueError, match="No valid YAML frontmatter"):
            parse_frontmatter(SAMPLE_MD_NO_FRONTMATTER)

    def test_empty_aliases_list_serialised_to_empty_string(self):
        """An empty aliases list must become an empty string."""
        metadata, _ = parse_frontmatter(SAMPLE_MD_NULL_FIELD)
        assert metadata["aliases"] == ""


# ---------------------------------------------------------------------------
# chunk_by_section
# ---------------------------------------------------------------------------


class TestChunkBySection:
    def _get_body(self) -> str:
        _, body = parse_frontmatter(SAMPLE_MD)
        return body

    def _get_metadata(self) -> dict:
        metadata, _ = parse_frontmatter(SAMPLE_MD)
        return metadata

    def test_happy_path_returns_one_document_per_section(self):
        """Body with 2 ## sections must produce exactly 2 Documents."""
        body = self._get_body()
        metadata = self._get_metadata()
        docs = chunk_by_section(body, metadata, source_stem="BHA_BHT")
        assert len(docs) == 2

    def test_metadata_attached_to_every_chunk(self):
        """All YAML metadata fields + source must appear in each chunk's metadata."""
        body = self._get_body()
        metadata = self._get_metadata()
        docs = chunk_by_section(body, metadata, source_stem="BHA_BHT")

        for doc in docs:
            assert doc.metadata["ingredient"] == "BHA_BHT"
            assert doc.metadata["source"] == "BHA_BHT"
            assert "risk_level" in doc.metadata

    def test_empty_chunks_are_skipped(self):
        """Trailing whitespace / blank sections must not produce empty Documents."""
        body = "\n\n## Summary\nSome content.\n\n\n"
        docs = chunk_by_section(body, {}, source_stem="test")
        assert all(doc.page_content.strip() for doc in docs)

    def test_single_section_returns_one_document(self):
        """A body with only one ## heading must return exactly 1 Document."""
        body = "## Only Section\nContent here.\n"
        docs = chunk_by_section(body, {}, source_stem="single")
        assert len(docs) == 1
        assert "Only Section" in docs[0].page_content

    def test_chunk_content_includes_heading(self):
        """Each chunk must preserve its ## heading as part of the content."""
        body = self._get_body()
        metadata = self._get_metadata()
        docs = chunk_by_section(body, metadata, source_stem="BHA_BHT")

        headings = [d.page_content for d in docs]
        assert any("## Summary" in h for h in headings)
        assert any("## Health Risks" in h for h in headings)


# ---------------------------------------------------------------------------
# load_md_files — uses pytest tmp_path fixture for real temp files
# ---------------------------------------------------------------------------


class TestLoadMdFiles:
    def test_happy_path_loads_two_files(self, tmp_path: Path):
        """Two valid .md files must be loaded; file_count must equal 2."""
        (tmp_path / "ingredient_a.md").write_text(SAMPLE_MD, encoding="utf-8")
        (tmp_path / "ingredient_b.md").write_text(SAMPLE_MD_NULL_FIELD, encoding="utf-8")

        docs, file_count = load_md_files(tmp_path)

        assert file_count == 2
        assert len(docs) > 0

    def test_malformed_file_is_skipped(self, tmp_path: Path):
        """A file without frontmatter must be skipped; valid file still processed."""
        (tmp_path / "valid.md").write_text(SAMPLE_MD, encoding="utf-8")
        (tmp_path / "broken.md").write_text(SAMPLE_MD_NO_FRONTMATTER, encoding="utf-8")

        docs, file_count = load_md_files(tmp_path)

        assert file_count == 1
        assert len(docs) > 0

    def test_empty_directory_returns_empty_results(self, tmp_path: Path):
        """A directory with no .md files must return ([], 0)."""
        docs, file_count = load_md_files(tmp_path)

        assert docs == []
        assert file_count == 0

    def test_source_metadata_matches_filename_stem(self, tmp_path: Path):
        """Each chunk's 'source' metadata field must equal the .md filename stem."""
        (tmp_path / "BHA_BHT.md").write_text(SAMPLE_MD, encoding="utf-8")

        docs, _ = load_md_files(tmp_path)

        for doc in docs:
            assert doc.metadata["source"] == "BHA_BHT"
