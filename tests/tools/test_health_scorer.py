"""Tests for tools/health_scorer.py.

Covers: clean product, sugar/salt/sat-fat penalties, controversial ingredients,
E-number detection, additive-free bonuses, score clamping, grade boundaries,
recommendation mapping, and output structure.
No mocking needed — pure heuristic logic.
"""

from __future__ import annotations

import pytest

from tools.health_scorer import score_health

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_CHICKEN_BREAST = {
    "ingredients": "chicken breast",
    "calories_per_100g": 165.0,
    "fat_per_100g": 3.6,
    "saturated_fat_per_100g": 1.0,
    "sugar_per_100g": 0.0,
    "salt_per_100g": 0.07,
    "fibre_per_100g": 0.0,
    "protein_per_100g": 31.0,
}

_HIGH_SUGAR_PRODUCT = {
    "ingredients": "sugar, glucose syrup, cocoa butter",
    "calories_per_100g": 500.0,
    "fat_per_100g": 20.0,
    "saturated_fat_per_100g": 12.0,
    "sugar_per_100g": 56.0,   # well above 22.5g severe threshold
    "salt_per_100g": 0.1,
    "fibre_per_100g": 0.0,
    "protein_per_100g": 3.0,
}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_clean_high_protein_product_grades_well():
    """Chicken breast: high protein, short list, no additives → A or B."""
    result = score_health.invoke(_CHICKEN_BREAST)
    assert result["grade"] in ("A", "B")
    assert any("protein" in p for p in result["positives"])


def test_no_additives_bonus_applied():
    """Clean ingredient list with no E-numbers → 'no food additives' in positives."""
    result = score_health.invoke(_CHICKEN_BREAST)
    assert any("no food additives" in p for p in result["positives"])


# ---------------------------------------------------------------------------
# Penalty cases
# ---------------------------------------------------------------------------


def test_high_sugar_penalty():
    """Very high sugar (>22.5g/100g) must appear in negatives and lower the grade."""
    result = score_health.invoke(_HIGH_SUGAR_PRODUCT)
    assert any("sugar" in n for n in result["negatives"])
    assert result["grade"] not in ("A", "B")


def test_controversial_ingredient_penalty():
    """Palm oil in ingredients must appear in negatives."""
    payload = {**_CHICKEN_BREAST, "ingredients": "chicken breast, palm oil"}
    result = score_health.invoke(payload)
    assert any("palm oil" in n for n in result["negatives"])


def test_e_number_penalty():
    """Multiple E-numbers should trigger an additive penalty entry."""
    payload = {
        **_HIGH_SUGAR_PRODUCT,
        "ingredients": "sugar, E471, E322, E420, artificial flavour",
    }
    result = score_health.invoke(payload)
    assert any("additive" in n for n in result["negatives"])
    assert result["score"] < 50


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------


def test_score_never_below_zero():
    """Even with all possible penalties the score must not go below 0."""
    worst_case = {
        "ingredients": "palm oil, BHA (E320), BHT (E321), E471, E433, E466, E951, carrageenan",
        "calories_per_100g": 600.0,
        "fat_per_100g": 35.0,
        "saturated_fat_per_100g": 15.0,
        "sugar_per_100g": 50.0,
        "salt_per_100g": 3.0,
        "fibre_per_100g": 0.0,
        "protein_per_100g": 2.0,
    }
    result = score_health.invoke(worst_case)
    assert result["score"] >= 0


def test_score_never_above_100():
    """Even with all possible bonuses the score must not exceed 100."""
    best_case = {
        "ingredients": "oats, whey protein",
        "calories_per_100g": 100.0,
        "fat_per_100g": 1.0,
        "saturated_fat_per_100g": 0.3,
        "sugar_per_100g": 1.0,
        "salt_per_100g": 0.05,
        "fibre_per_100g": 10.0,
        "protein_per_100g": 30.0,
    }
    result = score_health.invoke(best_case)
    assert result["score"] <= 100


# ---------------------------------------------------------------------------
# Grade boundaries
# ---------------------------------------------------------------------------


def test_grade_a_for_excellent_product():
    """A clean, high-protein, high-fibre product should earn grade A."""
    result = score_health.invoke(
        {
            "ingredients": "oats, whey protein",
            "calories_per_100g": 100.0,
            "fat_per_100g": 1.0,
            "saturated_fat_per_100g": 0.3,
            "sugar_per_100g": 1.0,
            "salt_per_100g": 0.05,
            "fibre_per_100g": 10.0,
            "protein_per_100g": 30.0,
        }
    )
    assert result["grade"] == "A"


def test_grade_f_for_worst_product():
    """Heavily penalised product should reach grade D or F."""
    worst_case = {
        "ingredients": "palm oil, BHA (E320), BHT (E321), E471, E433, E466, E951, carrageenan",
        "calories_per_100g": 600.0,
        "fat_per_100g": 35.0,
        "saturated_fat_per_100g": 15.0,
        "sugar_per_100g": 50.0,
        "salt_per_100g": 3.0,
        "fibre_per_100g": 0.0,
        "protein_per_100g": 2.0,
    }
    result = score_health.invoke(worst_case)
    assert result["grade"] in ("D", "F")


# ---------------------------------------------------------------------------
# Recommendation mapping
# ---------------------------------------------------------------------------


def test_recommendation_good_choice_for_high_grades():
    result = score_health.invoke(_CHICKEN_BREAST)
    assert result["recommendation"] == "good choice"


def test_recommendation_avoid_for_low_grades():
    worst_case = {
        "ingredients": "palm oil, BHA (E320), BHT (E321), E471, E433, E466",
        "calories_per_100g": 600.0,
        "fat_per_100g": 35.0,
        "saturated_fat_per_100g": 15.0,
        "sugar_per_100g": 50.0,
        "salt_per_100g": 3.0,
        "fibre_per_100g": 0.0,
        "protein_per_100g": 2.0,
    }
    result = score_health.invoke(worst_case)
    assert result["recommendation"] == "avoid"


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_output_structure():
    """Result must always contain all expected keys with correct types."""
    result = score_health.invoke(_CHICKEN_BREAST)
    assert "score" in result
    assert "grade" in result
    assert "positives" in result
    assert "negatives" in result
    assert "recommendation" in result
    assert isinstance(result["score"], float)
    assert result["grade"] in ("A", "B", "C", "D", "F")
    assert isinstance(result["positives"], list)
    assert isinstance(result["negatives"], list)
    assert isinstance(result["recommendation"], str)
