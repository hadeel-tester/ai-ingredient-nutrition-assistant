"""profile.py — User profile page.

Renders a form for the user to set personal preferences that personalise
every AI response: name, EU top-14 allergens, dietary preferences, health
goals, and daily calorie target.

Profile is persisted to data/user_profile.json. On page load the saved
profile is read so the form pre-fills with the user's existing settings.

load_user_profile() is also imported by app.py to pass the profile into
the chain at invoke time — keeping the chain stateless and cacheable.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EU_ALLERGENS: list[str] = [
    "Celery",
    "Cereals with gluten",
    "Crustaceans",
    "Eggs",
    "Fish",
    "Lupin",
    "Milk",
    "Molluscs",
    "Mustard",
    "Nuts",
    "Peanuts",
    "Sesame",
    "Soybeans",
    "Sulphites",
]

DIETARY_PREFERENCES: list[str] = [
    "Vegan",
    "Vegetarian",
    "Diabetic",
    "Low-sugar",
    "Low-salt",
    "Low-fat",
    "Gluten-free",
    "Lactose-free",
]

HEALTH_GOALS: list[str] = [
    "Avoid additives",
    "Avoid palm oil",
    "Reduce sugar",
    "Reduce salt",
    "Avoid controversial ingredients",
    "High protein",
    "High fibre",
]

_PROFILE_PATH: Path = Path("data/user_profile.json")

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def load_user_profile() -> dict:
    """Load the user profile from disk.

    Returns an empty dict if the file does not exist or is corrupt.

    Returns:
        Parsed profile dict with keys: name, allergens, preferences,
        goals, calorie_target. Returns {} if no profile is saved yet.
    """
    if not _PROFILE_PATH.exists():
        return {}
    try:
        return json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_user_profile(profile: dict) -> None:
    """Persist the user profile to disk as JSON.

    Creates the data/ directory if it does not exist.

    Args:
        profile: Dict with keys name, allergens, preferences, goals,
                 calorie_target.
    """
    _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PROFILE_PATH.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def render_profile_page() -> None:
    """Render the user profile form."""
    st.header("👤 Your Profile")
    st.caption(
        "Set your allergens, dietary preferences, and health goals. "
        "The assistant will factor these into every response and proactively "
        "warn you when a product or ingredient conflicts with your profile."
    )

    profile = load_user_profile()

    with st.form("profile_form"):
        name = st.text_input(
            "Your name",
            value=profile.get("name", ""),
            placeholder="e.g. Alice",
        )

        st.divider()

        allergens: list[str] = st.multiselect(
            "Allergens (EU top-14)",
            options=EU_ALLERGENS,
            default=profile.get("allergens", []),
            help="Select any allergens you need to avoid. "
                 "The AI will warn you when these appear in a product.",
        )

        preferences: list[str] = st.multiselect(
            "Dietary preferences",
            options=DIETARY_PREFERENCES,
            default=profile.get("preferences", []),
        )

        goals: list[str] = st.multiselect(
            "Health goals",
            options=HEALTH_GOALS,
            default=profile.get("goals", []),
        )

        st.divider()

        calorie_target: int = st.number_input(
            "Daily calorie target (kcal)",
            min_value=500,
            max_value=5000,
            value=profile.get("calorie_target", 2000),
            step=50,
            help="Used by the Nutrition Calculator to show % of your daily target.",
        )

        submitted = st.form_submit_button("💾 Save profile", use_container_width=True)

    if submitted:
        save_user_profile(
            {
                "name": name,
                "allergens": allergens,
                "preferences": preferences,
                "goals": goals,
                "calorie_target": int(calorie_target),
            }
        )
        st.success(
            "Profile saved! Your next chat message will use your updated profile."
        )
