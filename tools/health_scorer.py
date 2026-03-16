"""health_scorer.py — LangChain tool for scoring a product's overall health quality.

Produces a 0–100 score and A–F grade based on nutrient profile (per 100g) and
ingredient quality signals: E-number count and presence of controversial ingredients
such as palm oil, BHA/BHT, HFCS, and emulsifiers.

No external API calls — pure heuristic logic. Not medical advice.
"""

from __future__ import annotations

import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# Controversial ingredients matched as substrings in the lowercased ingredients text.
# Aliases for the same substance are grouped; any match flags that substance.
CONTROVERSIAL_INGREDIENTS: list[str] = [
    "palm oil",
    "high fructose corn syrup",
    "hfcs",
    "aspartame",
    "e951",
    "bha",
    "butylated hydroxyanisole",
    "e320",
    "bht",
    "butylated hydroxytoluene",
    "e321",
    "carrageenan",
    "e407",
    "polysorbate 80",
    "e433",
    "carboxymethylcellulose",
    "carboxymethyl cellulose",
    "cmc",
    "e466",
]

# UK FSA per-100g nutrient thresholds used for scoring
_SAT_FAT_SEVERE   = 5.0    # g/100g
_SAT_FAT_MODERATE = 2.5
_SUGAR_SEVERE     = 22.5
_SUGAR_MODERATE   = 11.25
_SALT_SEVERE      = 1.5
_SALT_MODERATE    = 0.75
_FIBRE_HIGH       = 6.0
_FIBRE_MEDIUM     = 3.0
_PROTEIN_HIGH     = 20.0
_PROTEIN_MEDIUM   = 10.0

_SHORT_INGREDIENT_THRESHOLD = 5   # fewer than this → bonus

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class HealthScorerInput(BaseModel):
    """Input schema for the health scorer tool.

    All nutrient values are per 100g as printed on a food label.
    """

    ingredients: str = Field(
        description="Raw ingredient label text (comma-separated or freeform)."
    )
    calories_per_100g: float = Field(ge=0, description="Energy in kcal per 100g.")
    fat_per_100g: float = Field(ge=0, description="Total fat in g per 100g.")
    saturated_fat_per_100g: float = Field(ge=0, description="Saturated fat in g per 100g.")
    sugar_per_100g: float = Field(ge=0, description="Total sugars in g per 100g.")
    salt_per_100g: float = Field(ge=0, description="Salt in g per 100g.")
    fibre_per_100g: float = Field(ge=0, default=0.0, description="Dietary fibre in g per 100g.")
    protein_per_100g: float = Field(ge=0, default=0.0, description="Protein in g per 100g.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_e_numbers(text: str) -> list[str]:
    """Return a deduplicated list of E-numbers found in the ingredients text.

    Args:
        text: Lowercased ingredients string.

    Returns:
        List of distinct E-number strings (e.g. ['e471', 'e322']).
    """
    return list(set(re.findall(r"\be\d{3,4}\b", text)))


def _detect_controversial(text: str) -> list[str]:
    """Return a deduplicated list of controversial ingredient aliases found in text.

    Args:
        text: Lowercased ingredients string.

    Returns:
        List of matched alias strings from CONTROVERSIAL_INGREDIENTS.
    """
    return list({alias for alias in CONTROVERSIAL_INGREDIENTS if alias in text})


def _letter_grade(score: float) -> str:
    """Convert a 0–100 numeric score to an A–F letter grade.

    Args:
        score: Numeric health score (0–100).

    Returns:
        Letter grade string: 'A', 'B', 'C', 'D', or 'F'.
    """
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    if score >= 20:
        return "D"
    return "F"


def _recommendation(grade: str) -> str:
    """Map a letter grade to a consumer recommendation.

    Args:
        grade: Letter grade string.

    Returns:
        One of 'good choice', 'consume in moderation', or 'avoid'.
    """
    if grade in ("A", "B"):
        return "good choice"
    if grade == "C":
        return "consume in moderation"
    return "avoid"


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(args_schema=HealthScorerInput)
def score_health(
    ingredients: str,
    calories_per_100g: float,
    fat_per_100g: float,
    saturated_fat_per_100g: float,
    sugar_per_100g: float,
    salt_per_100g: float,
    fibre_per_100g: float = 0.0,
    protein_per_100g: float = 0.0,
) -> dict:
    """Score a food product's overall health quality on a 0–100 scale.

    Combines nutrient profile (saturated fat, sugar, salt, fibre, protein)
    with ingredient quality signals (E-number count, controversial additives)
    to produce a holistic score, A–F grade, and consumer recommendation.

    Scoring is additive from a neutral base of 50:
    - Penalties: high saturated fat / sugar / salt, E-numbers, controversial ingredients
    - Bonuses: high fibre / protein, short clean ingredient list, no additives

    This is a heuristic model — not medical or regulatory advice.

    Args:
        ingredients:            Raw ingredients text from a product label.
        calories_per_100g:      Energy in kcal per 100g.
        fat_per_100g:           Total fat in g per 100g.
        saturated_fat_per_100g: Saturated fat in g per 100g.
        sugar_per_100g:         Total sugars in g per 100g.
        salt_per_100g:          Salt in g per 100g.
        fibre_per_100g:         Dietary fibre in g per 100g (default 0).
        protein_per_100g:       Protein in g per 100g (default 0).

    Returns:
        Dict with keys:
          - score:          numeric score 0–100 (1 decimal place)
          - grade:          letter grade A/B/C/D/F
          - positives:      list of factors that improved the score
          - negatives:      list of factors that reduced the score
          - recommendation: 'good choice', 'consume in moderation', or 'avoid'
    """
    text = ingredients.lower()
    e_numbers = _detect_e_numbers(text)
    controversial = _detect_controversial(text)
    ingredient_count = len([i for i in ingredients.split(",") if i.strip()])

    score = 50.0
    positives: list[str] = []
    negatives: list[str] = []

    # --- Penalties ---

    if saturated_fat_per_100g > _SAT_FAT_SEVERE:
        score -= 15
        negatives.append(f"very high saturated fat ({saturated_fat_per_100g}g/100g)")
    elif saturated_fat_per_100g > _SAT_FAT_MODERATE:
        score -= 8
        negatives.append(f"high saturated fat ({saturated_fat_per_100g}g/100g)")

    if sugar_per_100g > _SUGAR_SEVERE:
        score -= 15
        negatives.append(f"very high sugar ({sugar_per_100g}g/100g)")
    elif sugar_per_100g > _SUGAR_MODERATE:
        score -= 8
        negatives.append(f"high sugar ({sugar_per_100g}g/100g)")

    if salt_per_100g > _SALT_SEVERE:
        score -= 15
        negatives.append(f"very high salt ({salt_per_100g}g/100g)")
    elif salt_per_100g > _SALT_MODERATE:
        score -= 8
        negatives.append(f"high salt ({salt_per_100g}g/100g)")

    if e_numbers:
        e_penalty = min(len(e_numbers) * 3, 15)
        score -= e_penalty
        negatives.append(f"{len(e_numbers)} food additive(s) detected: {', '.join(sorted(e_numbers))}")

    if controversial:
        c_penalty = min(len(controversial) * 8, 24)
        score -= c_penalty
        negatives.append(f"controversial ingredient(s): {', '.join(sorted(controversial))}")

    # --- Bonuses ---

    if fibre_per_100g >= _FIBRE_HIGH:
        score += 15
        positives.append(f"high fibre ({fibre_per_100g}g/100g)")
    elif fibre_per_100g >= _FIBRE_MEDIUM:
        score += 8
        positives.append(f"good fibre content ({fibre_per_100g}g/100g)")

    if protein_per_100g >= _PROTEIN_HIGH:
        score += 15
        positives.append(f"high protein ({protein_per_100g}g/100g)")
    elif protein_per_100g >= _PROTEIN_MEDIUM:
        score += 8
        positives.append(f"good protein content ({protein_per_100g}g/100g)")

    if ingredient_count > 0 and ingredient_count < _SHORT_INGREDIENT_THRESHOLD:
        score += 10
        positives.append(f"short ingredient list ({ingredient_count} ingredients)")

    if not e_numbers and not controversial:
        score += 8
        positives.append("no food additives detected")

    if not controversial:
        score += 7
        positives.append("no controversial ingredients")

    score = round(max(0.0, min(100.0, score)), 1)
    grade = _letter_grade(score)

    return {
        "score":          score,
        "grade":          grade,
        "positives":      positives,
        "negatives":      negatives,
        "recommendation": _recommendation(grade),
    }
