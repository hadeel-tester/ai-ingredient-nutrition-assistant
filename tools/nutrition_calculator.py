"""nutrition_calculator.py — LangChain tool for per-serving nutrition evaluation.

Given per-100g nutrient values and a serving size, calculates per-serving amounts,
compares them against EU Daily Reference Intakes, and issues UK FSA traffic light
ratings per nutrient.

No external API calls — pure calculation.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# EU reference daily intakes (Regulation EU 1169/2011, Annex XIII)
EU_DRI: dict[str, float] = {
    "calories":       2000.0,  # kcal
    "fat":              70.0,  # g
    "saturated_fat":    20.0,  # g
    "sugar":            90.0,  # g
    "protein":          50.0,  # g
    "salt":              6.0,  # g
    "fibre":            25.0,  # g
}

# UK FSA traffic light thresholds — nutrients where LOWER per serving is better.
# Tuple: (green_ceiling, red_floor) — ≤ green_ceiling → green, > red_floor → red.
_FSA_LOW_IS_BETTER: dict[str, tuple[float, float]] = {
    "fat":           (3.0,  20.0),
    "saturated_fat": (1.5,   6.0),
    "sugar":         (5.0,  27.0),
    "salt":          (0.3,   2.4),
}

# UK FSA traffic light thresholds — nutrients where HIGHER per serving is better.
# Tuple: (red_ceiling, green_floor) — < red_ceiling → red, ≥ green_floor → green.
_FSA_HIGH_IS_BETTER: dict[str, tuple[float, float]] = {
    "protein": (5.0, 10.0),
    "fibre":   (3.0,  6.0),
}

# Calorie traffic light based on % of daily reference intake per serving.
# (green_ceiling_pct, red_floor_pct)
_CALORIE_DRI_BANDS: tuple[float, float] = (10.0, 25.0)

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class NutritionInput(BaseModel):
    """Input schema for the nutrition evaluator tool.

    All nutrient values are per 100g, as printed on a food label.
    """

    calories_per_100g: float = Field(ge=0, description="Energy in kcal per 100g.")
    fat_per_100g: float = Field(ge=0, description="Total fat in g per 100g.")
    saturated_fat_per_100g: float = Field(ge=0, description="Saturated fat in g per 100g.")
    sugar_per_100g: float = Field(ge=0, description="Total sugars in g per 100g.")
    protein_per_100g: float = Field(ge=0, description="Protein in g per 100g.")
    salt_per_100g: float = Field(ge=0, description="Salt in g per 100g.")
    fibre_per_100g: float = Field(ge=0, default=0.0, description="Dietary fibre in g per 100g.")
    serving_size_g: float = Field(
        gt=0, default=100.0, description="Serving size in grams (defaults to 100g)."
    )
    calorie_target: float | None = Field(
        default=None,
        description=(
            "User's daily calorie target in kcal from their profile. "
            "Pass this if the user profile contains a calorie_target. "
            "Scales DRI percentages to the user's personal intake goal "
            "instead of the standard 2000 kcal EU reference."
        ),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _effective_dri(calorie_target: float | None) -> dict[str, float]:
    """Return DRI reference values scaled to calorie_target, or EU_DRI if not set.

    Calories, fat, saturated_fat, sugar, protein, and salt are scaled
    proportionally to the ratio of calorie_target to the 2000 kcal EU default.
    Fibre is held fixed at 25g — it is not calorie-dependent.

    Args:
        calorie_target: User's daily calorie goal in kcal, or None for EU default.

    Returns:
        Dict with the same keys as EU_DRI, scaled to the target.
    """
    if not calorie_target or calorie_target == EU_DRI["calories"]:
        return EU_DRI
    ratio = calorie_target / EU_DRI["calories"]
    return {
        "calories":      calorie_target,
        "fat":           round(EU_DRI["fat"]           * ratio, 1),
        "saturated_fat": round(EU_DRI["saturated_fat"] * ratio, 1),
        "sugar":         round(EU_DRI["sugar"]         * ratio, 1),
        "protein":       round(EU_DRI["protein"]       * ratio, 1),
        "salt":          round(EU_DRI["salt"]          * ratio, 1),
        "fibre":         EU_DRI["fibre"],  # not calorie-dependent
    }


def _traffic_light(nutrient: str, value: float, dri_pct: float) -> str:
    """Return the UK FSA traffic light colour for a single nutrient.

    Args:
        nutrient:  Nutrient key matching the threshold dicts.
        value:     Per-serving amount (grams or kcal).
        dri_pct:   Percentage of EU daily reference intake per serving.

    Returns:
        One of 'green', 'amber', or 'red'.
    """
    if nutrient == "calories":
        green_ceil, red_floor = _CALORIE_DRI_BANDS
        if dri_pct <= green_ceil:
            return "green"
        if dri_pct > red_floor:
            return "red"
        return "amber"

    if nutrient in _FSA_LOW_IS_BETTER:
        green_ceil, red_floor = _FSA_LOW_IS_BETTER[nutrient]
        if value <= green_ceil:
            return "green"
        if value > red_floor:
            return "red"
        return "amber"

    if nutrient in _FSA_HIGH_IS_BETTER:
        red_ceil, green_floor = _FSA_HIGH_IS_BETTER[nutrient]
        if value >= green_floor:
            return "green"
        if value < red_ceil:
            return "red"
        return "amber"

    return "amber"  # fallback for any unknown nutrient


def _overall_rating(traffic_lights: dict[str, str]) -> str:
    """Derive an overall nutritional rating from the individual traffic lights.

    Args:
        traffic_lights: Dict mapping nutrient name → 'green'|'amber'|'red'.

    Returns:
        One of 'excellent', 'good', 'moderate', or 'poor'.
    """
    red_count = sum(1 for v in traffic_lights.values() if v == "red")
    green_count = sum(1 for v in traffic_lights.values() if v == "green")

    if red_count == 0 and green_count >= 5:
        return "excellent"
    if red_count == 0:
        return "good"
    if red_count <= 2:
        return "moderate"
    return "poor"


def _build_summary(overall: str, traffic_lights: dict[str, str]) -> str:
    """Build a human-readable summary sentence from the rating and traffic lights.

    Args:
        overall:       Overall rating string.
        traffic_lights: Dict mapping nutrient name → traffic light colour.

    Returns:
        A plain-English summary string.
    """
    if overall == "excellent":
        return "Excellent nutritional profile per serving."

    reds = [n.replace("_", " ") for n, v in traffic_lights.items() if v == "red"]
    greens = [n.replace("_", " ") for n, v in traffic_lights.items() if v == "green"]

    parts: list[str] = [f"{overall.capitalize()} nutritional profile per serving."]
    if reds:
        parts.append(f"High in: {', '.join(reds)}.")
    if greens:
        parts.append(f"Good source of: {', '.join(greens)}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(args_schema=NutritionInput)
def evaluate_nutrition(
    calories_per_100g: float,
    fat_per_100g: float,
    saturated_fat_per_100g: float,
    sugar_per_100g: float,
    protein_per_100g: float,
    salt_per_100g: float,
    fibre_per_100g: float = 0.0,
    serving_size_g: float = 100.0,
    calorie_target: float | None = None,
) -> dict:
    """Evaluate the nutritional quality of a food product per serving.

    Converts per-100g label values to per-serving amounts, calculates each
    nutrient as a percentage of the EU Daily Reference Intake, and assigns a
    UK FSA traffic light colour (green / amber / red) to each nutrient.
    For nutrients where more is better (protein, fibre) the colour logic is
    inverted relative to fat, sugar, and salt.

    Args:
        calories_per_100g:      Energy in kcal per 100g.
        fat_per_100g:           Total fat in g per 100g.
        saturated_fat_per_100g: Saturated fat in g per 100g.
        sugar_per_100g:         Total sugars in g per 100g.
        protein_per_100g:       Protein in g per 100g.
        salt_per_100g:          Salt in g per 100g.
        fibre_per_100g:         Dietary fibre in g per 100g (default 0).
        serving_size_g:         Serving size in grams (default 100).
        calorie_target:         User's daily calorie goal in kcal. If provided,
                                DRI percentages are scaled to this target instead
                                of the standard 2000 kcal EU reference.

    Returns:
        Dict with keys:
          - serving_size_g: the serving size used
          - per_serving: per-serving amounts for each nutrient
          - dri_percent: % of daily reference intake per serving
          - traffic_lights: green/amber/red rating per nutrient
          - overall: overall rating (excellent/good/moderate/poor)
          - summary: human-readable assessment sentence
          - reference: DRI reference used (EU DRI or user profile)
    """
    ratio = serving_size_g / 100.0

    raw_per_100g: dict[str, float] = {
        "calories":       calories_per_100g,
        "fat":            fat_per_100g,
        "saturated_fat":  saturated_fat_per_100g,
        "sugar":          sugar_per_100g,
        "protein":        protein_per_100g,
        "salt":           salt_per_100g,
        "fibre":          fibre_per_100g,
    }

    per_serving: dict[str, float] = {
        nutrient: round(value * ratio, 2)
        for nutrient, value in raw_per_100g.items()
    }

    effective_dri = _effective_dri(calorie_target)
    dri_percent: dict[str, float] = {
        nutrient: round((per_serving[nutrient] / effective_dri[nutrient]) * 100, 1)
        for nutrient in per_serving
    }

    traffic_lights: dict[str, str] = {
        nutrient: _traffic_light(nutrient, per_serving[nutrient], dri_percent[nutrient])
        for nutrient in per_serving
    }

    overall = _overall_rating(traffic_lights)
    summary = _build_summary(overall, traffic_lights)

    reference = (
        f"User profile ({int(calorie_target)} kcal)"
        if calorie_target and calorie_target != EU_DRI["calories"]
        else "EU DRI (2000 kcal)"
    )

    return {
        "serving_size_g":  serving_size_g,
        "per_serving":     per_serving,
        "dri_percent":     dri_percent,
        "traffic_lights":  traffic_lights,
        "overall":         overall,
        "summary":         summary,
        "reference":       reference,
    }
