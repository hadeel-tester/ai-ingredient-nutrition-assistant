"""Tests for tools/ingredient_analyzer.py.

All tests mock the ChromaDB vector store — no real embeddings or API calls.
Covers: ingredient parsing, KB matching, tool happy path, no-match fallback,
and edge cases (empty input, duplicate ingredients).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from tools.ingredient_analyzer import (
    _parse_ingredients,
    _query_kb_for_ingredient,
    analyze_product_ingredients,
)


# ---------------------------------------------------------------------------
# Ingredient parsing
# ---------------------------------------------------------------------------


def test_parse_basic_comma_list():
    """Standard comma-separated ingredients are split and lowercased."""
    result = _parse_ingredients("Sugar, Palm Oil, Cocoa, Milk Powder")
    assert result == ["sugar", "palm oil", "cocoa", "milk powder"]


def test_parse_strips_percentages():
    """Percentage values (e.g. '10%') are removed from ingredient names."""
    result = _parse_ingredients("sugar 10%, cocoa 5.5%, palm oil")
    assert "sugar" in result
    assert "cocoa" in result
    assert "10" not in " ".join(result)


def test_parse_removes_parenthetical_content():
    """Parenthetical sub-ingredients like '(soy)' are removed."""
    result = _parse_ingredients("lecithin (soy), sugar, flavouring (artificial)")
    assert "lecithin" in result
    assert "soy" not in result
    assert "flavouring" in result


def test_parse_deduplicates():
    """Duplicate ingredient names appear only once."""
    result = _parse_ingredients("sugar, salt, sugar, salt")
    assert result.count("sugar") == 1
    assert result.count("salt") == 1


def test_parse_empty_string():
    """Empty input returns empty list."""
    assert _parse_ingredients("") == []


def test_parse_semicolons_and_dots():
    """Semicolons and periods also work as separators."""
    result = _parse_ingredients("sugar; salt. pepper")
    assert len(result) == 3


# ---------------------------------------------------------------------------
# KB matching
# ---------------------------------------------------------------------------


def _make_mock_vectorstore(docs_with_scores: list[tuple]) -> MagicMock:
    """Create a mock vectorstore returning the given (doc, score) pairs."""
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = docs_with_scores
    return mock_vs


def _make_doc(ingredient: str, aliases: str = "", section: str = "Summary",
              risk_level: str = "low", content: str = "Test content.") -> MagicMock:
    """Create a mock Document with metadata."""
    doc = MagicMock()
    doc.metadata = {
        "ingredient": ingredient,
        "aliases": aliases,
        "section": section,
        "risk_level": risk_level,
        "category": "test",
    }
    doc.page_content = content
    return doc


def test_query_kb_match_by_name():
    """Ingredient matching its KB name returns results."""
    doc = _make_doc("Aspartame", aliases="E951, NutraSweet")
    vs = _make_mock_vectorstore([(doc, 0.2)])

    result = _query_kb_for_ingredient("aspartame", vs)
    assert result is not None
    assert len(result) == 1
    assert result[0]["ingredient"] == "Aspartame"


def test_query_kb_match_by_alias():
    """Ingredient matching a KB alias returns results."""
    doc = _make_doc("Aspartame", aliases="E951, NutraSweet")
    vs = _make_mock_vectorstore([(doc, 0.3)])

    result = _query_kb_for_ingredient("nutrasweet", vs)
    assert result is not None
    assert result[0]["ingredient"] == "Aspartame"


def test_query_kb_no_match():
    """Ingredient with no KB entry returns None."""
    doc = _make_doc("Aspartame", aliases="E951")
    vs = _make_mock_vectorstore([(doc, 0.9)])

    result = _query_kb_for_ingredient("sugar", vs)
    assert result is None


# ---------------------------------------------------------------------------
# Full tool (mocked vectorstore)
# ---------------------------------------------------------------------------


def test_tool_happy_path(mocker):
    """Tool returns found ingredients and not_in_kb list."""
    aspartame_doc = _make_doc("Aspartame", aliases="E951", risk_level="high")
    palm_oil_doc = _make_doc("coconut_oil", aliases="coconut fat")

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.side_effect = [
        [(aspartame_doc, 0.2)],       # sugar → no match
        [(aspartame_doc, 0.1)],       # aspartame → match
        [(palm_oil_doc, 0.8)],        # salt → no match
    ]

    mocker.patch("tools.ingredient_analyzer.get_vectorstore", return_value=mock_vs)

    result = analyze_product_ingredients.invoke(
        {"ingredients_text": "sugar, aspartame, salt"}
    )

    assert "found" in result
    assert "not_in_kb" in result
    assert "summary" in result
    assert len(result["found"]) == 1
    assert result["found"][0]["ingredient"] == "Aspartame"
    assert "sugar" in result["not_in_kb"]
    assert "salt" in result["not_in_kb"]


def test_tool_no_ingredients(mocker):
    """Empty ingredients text returns an error."""
    result = analyze_product_ingredients.invoke({"ingredients_text": ""})
    assert "error" in result


def test_tool_deduplicates_kb_entries(mocker):
    """Same KB ingredient found via multiple product ingredients is listed once."""
    aspartame_doc = _make_doc("Aspartame", aliases="E951, NutraSweet")

    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.side_effect = [
        [(aspartame_doc, 0.1)],  # aspartame → match
        [(aspartame_doc, 0.1)],  # e951 → match (same KB entry)
    ]

    mocker.patch("tools.ingredient_analyzer.get_vectorstore", return_value=mock_vs)

    result = analyze_product_ingredients.invoke(
        {"ingredients_text": "aspartame, e951"}
    )

    # Should only appear once despite two input ingredients matching the same KB entry
    assert len(result["found"]) == 1
    assert result["found"][0]["ingredient"] == "Aspartame"
