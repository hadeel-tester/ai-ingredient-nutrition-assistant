"""Tests for tools/nutrition_calculator.py.

Covers: per-serving scaling, traffic light colours (green/amber/red) for
multiple nutrients, DRI percent calculation, overall rating logic,
default serving size, and output structure.
No mocking needed — pure arithmetic and threshold logic.
"""

from __future__ import annotations

import pytest

from tools.nutrition_calculator import evaluate_nutrition

# ---------------------------------------------------------------------------
# Shared base payload — clean product at 100g serving
# ---------------------------------------------------------------------------

_BASE = {
    "calories_per_100g": 100.0,
    "fat_per_100g": 1.0,
    "saturated_fat_per_100g": 0.5,
    "sugar_per_100g": 2.0,
    "protein_per_100g": 20.0,
    "salt_per_100g": 0.1,
    "fibre_per_100g": 8.0,
    "serving_size_g": 100.0,
}


# ---------------------------------------------------------------------------
# Happy path — per-serving scaling
# ---------------------------------------------------------------------------


def test_per_serving_scaling():
    """Per-serving values must equal per-100g × (serving / 100)."""
    result = evaluate_nutrition.invoke(
        {**_BASE, "calories_per_100g": 200.0, "serving_size_g": 15.0}
    )
    assert result["per_serving"]["calories"] == pytest.approx(30.0, rel=1e-3)


def test_serving_size_stored_in_output():
    """The serving_size_g used must be echoed back in the result."""
    result = evaluate_nutrition.invoke({**_BASE, "serving_size_g": 30.0})
    assert result["serving_size_g"] == 30.0


def test_default_serving_size_is_100g():
    """Omitting serving_size_g should default to 100g (per_serving == per_100g)."""
    payload = {k: v for k, v in _BASE.items() if k != "serving_size_g"}
    result = evaluate_nutrition.invoke(payload)
    assert result["serving_size_g"] == 100.0
    assert result["per_serving"]["calories"] == pytest.approx(
        payload["calories_per_100g"], rel=1e-3
    )


# ---------------------------------------------------------------------------
# Traffic light — low-is-better nutrients
# ---------------------------------------------------------------------------


def test_traffic_light_green_low_fat():
    """Fat ≤ 3g per serving → green."""
    result = evaluate_nutrition.invoke(
        {**_BASE, "fat_per_100g": 1.0, "serving_size_g": 100.0}
    )
    # 1.0g per serving → ≤ 3g → green
    assert result["traffic_lights"]["fat"] == "green"


def test_traffic_light_red_high_fat():
    """Fat > 20g per serving → red."""
    result = evaluate_nutrition.invoke(
        {**_BASE, "fat_per_100g": 25.0, "serving_size_g": 100.0}
    )
    # 25g per serving → > 20g → red
    assert result["traffic_lights"]["fat"] == "red"


def test_traffic_light_amber_fat():
    """Fat between 3g and 20g per serving → amber."""
    result = evaluate_nutrition.invoke(
        {**_BASE, "fat_per_100g": 10.0, "serving_size_g": 100.0}
    )
    # 10g per serving → amber
    assert result["traffic_lights"]["fat"] == "amber"


def test_traffic_light_red_high_sugar():
    """Sugar > 27g per serving → red."""
    result = evaluate_nutrition.invoke(
        {**_BASE, "sugar_per_100g": 30.0, "serving_size_g": 100.0}
    )
    assert result["traffic_lights"]["sugar"] == "red"


# ---------------------------------------------------------------------------
# Traffic light — high-is-better nutrients
# ---------------------------------------------------------------------------


def test_traffic_light_green_high_protein():
    """Protein ≥ 10g per serving → green."""
    result = evaluate_nutrition.invoke(
        {**_BASE, "protein_per_100g": 15.0, "serving_size_g": 100.0}
    )
    # 15g per serving → ≥ 10g → green
    assert result["traffic_lights"]["protein"] == "green"


def test_traffic_light_red_low_fibre():
    """Fibre < 3g per serving → red."""
    result = evaluate_nutrition.invoke(
        {**_BASE, "fibre_per_100g": 1.0, "serving_size_g": 100.0}
    )
    # 1g per serving → < 3g → red
    assert result["traffic_lights"]["fibre"] == "red"


# ---------------------------------------------------------------------------
# DRI percent calculation
# ---------------------------------------------------------------------------


def test_dri_percent_calories():
    """200 kcal / 2000 kcal DRI × 100 = 10.0%."""
    result = evaluate_nutrition.invoke(
        {**_BASE, "calories_per_100g": 200.0, "serving_size_g": 100.0}
    )
    assert result["dri_percent"]["calories"] == pytest.approx(10.0, rel=1e-3)


def test_dri_percent_protein():
    """20g protein / 50g DRI × 100 = 40.0%."""
    result = evaluate_nutrition.invoke(
        {**_BASE, "protein_per_100g": 20.0, "serving_size_g": 100.0}
    )
    assert result["dri_percent"]["protein"] == pytest.approx(40.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Overall rating
# ---------------------------------------------------------------------------


def test_overall_excellent_clean_product():
    """Clean low-fat, high-protein, high-fibre product → excellent or good."""
    result = evaluate_nutrition.invoke(_BASE)
    assert result["overall"] in ("excellent", "good")


def test_overall_poor_unhealthy_product():
    """Very high fat, sugar, salt → poor or moderate overall rating."""
    result = evaluate_nutrition.invoke(
        {
            "calories_per_100g": 600.0,
            "fat_per_100g": 35.0,
            "saturated_fat_per_100g": 15.0,
            "sugar_per_100g": 50.0,
            "protein_per_100g": 2.0,
            "salt_per_100g": 3.0,
            "fibre_per_100g": 0.0,
            "serving_size_g": 100.0,
        }
    )
    assert result["overall"] in ("moderate", "poor")


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_output_structure():
    """Result must contain all expected top-level and nested keys."""
    result = evaluate_nutrition.invoke(_BASE)

    assert "serving_size_g" in result
    assert "per_serving" in result
    assert "dri_percent" in result
    assert "traffic_lights" in result
    assert "overall" in result
    assert "summary" in result

    nutrients = {"calories", "fat", "saturated_fat", "sugar", "protein", "salt", "fibre"}
    assert set(result["per_serving"].keys()) == nutrients
    assert set(result["dri_percent"].keys()) == nutrients
    assert set(result["traffic_lights"].keys()) == nutrients

    for colour in result["traffic_lights"].values():
        assert colour in ("green", "amber", "red")

    assert result["overall"] in ("excellent", "good", "moderate", "poor")
    assert isinstance(result["summary"], str)
