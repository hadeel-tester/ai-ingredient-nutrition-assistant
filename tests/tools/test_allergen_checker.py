"""Tests for tools/allergen_checker.py.

Covers: definite matches, E-number aliases, uncommon name aliases,
possible-only detection, safe output, case insensitivity, and output structure.
No mocking needed — pure string matching.
"""

from __future__ import annotations

import pytest

from tools.allergen_checker import check_allergens


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_definite_match_common_names():
    """Common ingredient names should trigger confirmed allergen detection."""
    result = check_allergens.invoke(
        {"ingredients": "wheat flour, sugar, whey powder, palm oil"}
    )
    assert "cereals containing gluten" in result["detected"]
    assert "milk" in result["detected"]


def test_alias_e_number_triggers_definite_match():
    """E322 is listed as a definite alias for soybeans and must be detected."""
    result = check_allergens.invoke(
        {"ingredients": "sunflower oil, E322, salt, sugar"}
    )
    assert "soybeans" in result["detected"]


def test_alias_uncommon_names():
    """Uncommon aliases like tahini (sesame) and arachis oil (peanuts) must be caught."""
    result = check_allergens.invoke({"ingredients": "tahini, arachis oil, lemon juice"})
    assert "sesame" in result["detected"]
    assert "peanuts" in result["detected"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_possible_only_ambiguous_term():
    """'lecithin' alone is ambiguous — should appear in possible, not detected."""
    result = check_allergens.invoke(
        {"ingredients": "lecithin, glucose syrup, citric acid"}
    )
    assert result["detected"] == []
    assert len(result["possible"]) > 0
    # lecithin can flag soybeans or eggs as possible
    possible_set = set(result["possible"])
    assert possible_set & {"soybeans", "eggs"}


def test_safe_no_allergens():
    """A completely clean ingredient list should return empty detected and possible."""
    result = check_allergens.invoke({"ingredients": "water, salt, vinegar, spices"})
    assert result["detected"] == []
    assert result["possible"] == []
    assert "No EU" in result["message"]


def test_case_insensitive_detection():
    """Detection must be case-insensitive — uppercase input equals lowercase."""
    upper = check_allergens.invoke({"ingredients": "WHEAT FLOUR, EGG WHITE"})
    lower = check_allergens.invoke({"ingredients": "wheat flour, egg white"})
    assert set(upper["detected"]) == set(lower["detected"])


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_output_structure():
    """Result must always contain exactly the expected keys."""
    result = check_allergens.invoke({"ingredients": "water"})
    assert "detected" in result
    assert "possible" in result
    assert "message" in result
    assert isinstance(result["detected"], list)
    assert isinstance(result["possible"], list)
    assert isinstance(result["message"], str)
