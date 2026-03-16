"""allergen_checker.py — LangChain tool for detecting EU allergens in ingredient labels.

Given a raw ingredients string (comma-separated or freeform text), checks for all 14
EU-mandatory allergens using keyword matching, E-number recognition, and common aliases.
Returns confirmed detections, uncertain possibilities, and a human-readable message.

No external API calls — pure string matching against curated keyword maps.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Keyword maps
# ---------------------------------------------------------------------------

# Definite matches: any keyword here confirms the allergen is present.
DEFINITE_KEYWORDS: dict[str, list[str]] = {
    "celery": [
        "celery",
        "celeriac",
        "celery seed",
        "celery salt",
        "celery oil",
    ],
    "cereals containing gluten": [
        "wheat",
        "rye",
        "barley",
        "oats",
        "oat",
        "spelt",
        "kamut",
        "triticale",
        "farro",
        "durum",
        "semolina",
        "gluten",
        "bulgur",
        "couscous",
        "wheat starch",
        "wheat flour",
        "wheat bran",
        "wheat germ",
        "wheat protein",
        "barley malt",
        "rye flour",
        "spelt flour",
        "dinkel",
        "whole grain wheat",
    ],
    "crustaceans": [
        "crab",
        "lobster",
        "prawn",
        "shrimp",
        "crayfish",
        "langoustine",
        "scampi",
        "barnacle",
        "krill",
    ],
    "eggs": [
        "egg",
        "eggs",
        "egg white",
        "egg yolk",
        "albumin",
        "ovalbumin",
        "ovomucin",
        "meringue",
        "mayonnaise",
        "lysozyme",
        "e1105",
    ],
    "fish": [
        "fish",
        "cod",
        "haddock",
        "plaice",
        "salmon",
        "trout",
        "herring",
        "tuna",
        "mackerel",
        "anchovy",
        "anchovies",
        "sardine",
        "sprat",
        "pilchard",
        "bass",
        "carp",
        "eel",
        "perch",
        "pike",
        "tilapia",
        "fish sauce",
        "fish oil",
        "worcestershire sauce",
    ],
    "lupin": [
        "lupin",
        "lupine",
        "lupin flour",
        "lupin seed",
    ],
    "milk": [
        "milk",
        "dairy",
        "cream",
        "butter",
        "cheese",
        "yogurt",
        "yoghurt",
        "lactose",
        "casein",
        "whey",
        "ghee",
        "lactalbumin",
        "caseinate",
        "milk powder",
        "milk solids",
        "milk protein",
        "lactoferrin",
        "skimmed milk",
        "whole milk",
        "condensed milk",
        "milk fat",
    ],
    "molluscs": [
        "oyster",
        "mussel",
        "clam",
        "squid",
        "octopus",
        "scallop",
        "abalone",
        "snail",
        "cuttlefish",
        "whelk",
        "cockle",
        "periwinkle",
    ],
    "mustard": [
        "mustard",
        "mustard seed",
        "mustard oil",
        "mustard flour",
        "mustard leaves",
        "dijon",
        "wholegrain mustard",
    ],
    "peanuts": [
        "peanut",
        "groundnut",
        "monkey nut",
        "arachis oil",
        "peanut butter",
        "peanut oil",
    ],
    "sesame": [
        "sesame",
        "tahini",
        "til",
        "gingelly",
        "sesame oil",
        "sesame seed",
        "sesame flour",
        "benne",
    ],
    "soybeans": [
        "soy",
        "soya",
        "soybean",
        "tofu",
        "tempeh",
        "miso",
        "edamame",
        "soy sauce",
        "tamari",
        "soy milk",
        "soy protein",
        "soy flour",
        "soy oil",
        "soy lecithin",
        "soya lecithin",
        "textured vegetable protein",
        "tvp",
        "e322",
    ],
    "sulphur dioxide and sulphites": [
        "sulphite",
        "sulfite",
        "sulphur dioxide",
        "sulfur dioxide",
        "e220",
        "e221",
        "e222",
        "e223",
        "e224",
        "e225",
        "e226",
        "e227",
        "e228",
    ],
    "tree nuts": [
        "almond",
        "hazelnut",
        "walnut",
        "cashew",
        "pecan",
        "brazil nut",
        "pistachio",
        "macadamia",
        "queensland nut",
        "pine nut",
        "marzipan",
        "praline",
        "nougat",
    ],
}

# Possible matches: ambiguous terms that might indicate an allergen.
# An allergen only appears here if it was NOT already confirmed by DEFINITE_KEYWORDS.
POSSIBLE_KEYWORDS: dict[str, list[str]] = {
    "cereals containing gluten": [
        "malt",    # usually barley, but sometimes maize
        "starch",  # could be corn, potato, or wheat
        "flour",   # could be rice, corn, or gluten flour
    ],
    "eggs": [
        "lecithin",  # also from soy or sunflower
    ],
    "soybeans": [
        "lecithin",                      # also from egg or sunflower
        "vegetable protein",
        "hydrolysed vegetable protein",
    ],
    "fish": [
        "omega-3",
        "marine oil",
    ],
    "milk": [
        "caramel colour",  # sometimes dairy-derived
        "lactic acid",     # usually synthetic but can be dairy-derived
    ],
    "crustaceans": [
        "shellfish",
        "seafood",
    ],
    "molluscs": [
        "shellfish",
        "seafood",
    ],
    "peanuts": [
        "mixed nuts",
        "nut oil",
    ],
    "tree nuts": [
        "mixed nuts",
        "nut oil",
        "nut",
    ],
}


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


class AllergenCheckInput(BaseModel):
    """Input schema for the allergen checker tool."""

    ingredients: str = Field(
        description=(
            "Raw ingredients text from a food label. "
            "Can be comma-separated (e.g. 'wheat flour, sugar, whey, palm oil') "
            "or a plain description."
        )
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool(args_schema=AllergenCheckInput)
def check_allergens(ingredients: str) -> dict[str, list[str] | str]:
    """Identify EU-regulated allergens present in a food ingredient label.

    Checks all 14 EU mandatory allergens using curated keyword aliases and
    E-number codes. Returns confirmed detections, uncertain possibilities,
    and a human-readable summary message.

    Detected allergens are confirmed by specific keywords (e.g. 'whey' → milk,
    'tahini' → sesame, 'E322' → soybeans, 'E220' → sulphites).
    Possible allergens are flagged when only ambiguous terms appear
    (e.g. 'lecithin' could be soy, egg, or sunflower).

    Args:
        ingredients: Raw ingredients string from a product label.

    Returns:
        Dict with keys:
          - 'detected': list of confirmed EU allergens
          - 'possible': list of uncertain allergens (verify original label)
          - 'message': human-readable warning or safe confirmation
    """
    text = ingredients.lower().replace("gluten-free", "glutenfree")

    detected: list[str] = [
        allergen
        for allergen, keywords in DEFINITE_KEYWORDS.items()
        if any(kw in text for kw in keywords)
    ]

    possible: list[str] = [
        allergen
        for allergen, keywords in POSSIBLE_KEYWORDS.items()
        if allergen not in detected and any(kw in text for kw in keywords)
    ]

    if detected and possible:
        detected_str = ", ".join(detected)
        possible_str = ", ".join(possible)
        message = (
            f"WARNING: Allergens detected: {detected_str}. "
            f"Also uncertain: {possible_str} — verify the original label."
        )
    elif detected:
        message = f"WARNING: Allergens detected: {', '.join(detected)}."
    elif possible:
        possible_str = ", ".join(possible)
        message = (
            f"WARNING: Possible allergens (uncertain): {possible_str}. "
            "Verify the original product label."
        )
    else:
        message = "No EU-regulated allergens detected in these ingredients."

    return {"detected": detected, "possible": possible, "message": message}
