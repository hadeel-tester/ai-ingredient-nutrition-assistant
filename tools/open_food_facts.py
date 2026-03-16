"""open_food_facts.py — LangChain tool for live product lookup via Open Food Facts.

Given a product barcode or name, calls the Open Food Facts public API and returns
structured nutrition data: macros, allergens, Nutri-Score grade, and NOVA group.

No API key required. Uses the requests library with a 60-second timeout.
All extracted fields use .get() with a 'not available' fallback so the returned
dict never contains None — safe to display and pass to the LLM without null-checks.

Retry strategy: transient network failures (Timeout, ConnectionError) are retried
up to 3 times with exponential backoff (1 s, 2 s). HTTP errors and "not found"
responses are not retried — they reflect the actual state of the API.
"""

from __future__ import annotations

import requests
from urllib.parse import quote
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_OFF_BARCODE_URL = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
_OFF_SEARCH_URL = (
    "https://world.openfoodfacts.org/cgi/search.pl"
    "?search_terms={name}&search_simple=1&action=process&json=1"
    "&page_size=5"
    "&fields=product_name,product_name_en,ingredients_text,ingredients_text_en"
    ",allergens_tags,nutriments,nutriscore_grade,nova_group"
)
_REQUEST_TIMEOUT = 30  # seconds per attempt — retried up to 3 times on timeout
_NA = "not available"  # fallback for any missing field

# Retry only transient network failures (timeout / connection drop).
# HTTP errors (4xx/5xx) and "product not found" are not retried — they are
# deterministic responses from the server, not transient failures.
_RETRYABLE = (requests.exceptions.Timeout, requests.exceptions.ConnectionError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _get(url: str) -> requests.Response:
    """Perform a GET request with automatic retry on transient network errors.

    Retried up to 3 times with exponential backoff (1 s, 2 s, capped at 4 s).
    Only Timeout and ConnectionError trigger a retry; all other exceptions
    (including HTTP errors) propagate immediately.

    Args:
        url: Fully-formed request URL.

    Returns:
        requests.Response on success.

    Raises:
        requests.exceptions.Timeout: After all retry attempts are exhausted.
        requests.exceptions.ConnectionError: After all retry attempts are exhausted.
        requests.exceptions.RequestException: Immediately for non-retryable errors.
    """
    return requests.get(url, timeout=_REQUEST_TIMEOUT)


def _best_match(products: list[dict], query: str) -> dict:
    """Pick the product whose name best matches the search query.

    Scores each product by how many query words appear in its name (case-insensitive).
    Falls back to the first result when no name contains any query word, since the
    API already returns results in relevance order.

    Args:
        products: Non-empty list of product dicts from the API.
        query: Original search query string.

    Returns:
        The single best-matching product dict.
    """
    query_words = query.lower().split()

    def _score(product: dict) -> int:
        name = (product.get("product_name") or product.get("product_name_en") or "").lower()
        return sum(1 for w in query_words if w in name)

    return max(products, key=_score)


class OpenFoodFactsInput(BaseModel):
    """Input schema for the Open Food Facts lookup tool."""

    query: str = Field(
        description=(
            "Product barcode (digits only, e.g. '3017620422003') "
            "or product name (e.g. 'coca cola') to look up."
        )
    )


def _is_barcode(query: str) -> bool:
    """Return True if query looks like a barcode (all digit characters).

    Covers EAN-8, EAN-13, UPC-A, and similar numeric barcode formats.

    Args:
        query: Raw user input string.

    Returns:
        True if every character in the stripped query is a digit.
    """
    return query.strip().isdigit()


def _extract_product(product: dict) -> dict:
    """Normalise a raw Open Food Facts product dict into the standard return shape.

    All fields use .get() with a 'not available' fallback so the returned dict
    never contains None. Allergen tags are stripped of their language prefix
    (e.g. 'en:milk' → 'milk'); an empty allergen list becomes 'not available'.

    Ingredients prefer the English-specific field (ingredients_text_en) and fall
    back to the generic ingredients_text when the English one is absent.

    Args:
        product: Raw product dict from the Open Food Facts API response.

    Returns:
        Structured dict with product_name, ingredients, allergens, nutriments,
        nutriscore_grade, and nova_group.
    """
    nutriments = product.get("nutriments", {})

    raw_allergens: list[str] = product.get("allergens_tags", [])
    allergens = [tag.split(":", 1)[-1] for tag in raw_allergens] or _NA

    ingredients = (
        product.get("ingredients_text_en")
        or product.get("ingredients_text")
        or _NA
    )

    return {
        "product_name": (
            product.get("product_name")
            or product.get("product_name_en")
            or _NA
        ),
        "ingredients": ingredients,
        "allergens": allergens,
        "nutriments": {
            "calories_per_100g": nutriments.get("energy-kcal_100g", _NA),
            "fat_per_100g":      nutriments.get("fat_100g",          _NA),
            "sugar_per_100g":    nutriments.get("sugars_100g",       _NA),
            "protein_per_100g":  nutriments.get("proteins_100g",     _NA),
            "salt_per_100g":     nutriments.get("salt_100g",         _NA),
        },
        "nutriscore_grade": product.get("nutriscore_grade", _NA),
        "nova_group":       product.get("nova_group",       _NA),
    }


@tool(args_schema=OpenFoodFactsInput)
def lookup_product(query: str) -> dict:
    """Look up nutrition data for a food product using the Open Food Facts API.

    Accepts either a numeric barcode or a product name. Returns structured
    data including macronutrients, allergens, Nutri-Score grade, and NOVA group
    (food processing level 1–4).

    Args:
        query: Product barcode (digits only) or product name string.

    Returns:
        On success: dict with keys product_name, ingredients, allergens,
        nutriments (calories, fat, sugar, protein, salt per 100 g),
        nutriscore_grade, and nova_group.
        On failure: dict with a single 'error' key describing the problem.
    """
    try:
        if _is_barcode(query):
            url = _OFF_BARCODE_URL.format(barcode=query.strip())
            response = _get(url)

            if response.status_code != 200:
                return {"error": f"Unexpected HTTP {response.status_code} from API."}

            data = response.json()
            if data.get("status") == 0:
                return {"error": "Product not found for barcode."}

            return _extract_product(data["product"])

        else:
            url = _OFF_SEARCH_URL.format(name=quote(query.strip()))
            response = _get(url)

            if response.status_code != 200:
                return {"error": f"Unexpected HTTP {response.status_code} from API."}

            data = response.json()
            products = data.get("products", [])
            if not products:
                return {"error": "No product found for that name."}

            return _extract_product(_best_match(products, query))

    except requests.exceptions.Timeout:
        return {"error": f"Request timed out after {_REQUEST_TIMEOUT} s (after retries)."}
    except requests.exceptions.ConnectionError as exc:
        return {"error": f"Connection failed after retries: {exc}"}
    except requests.exceptions.RequestException as exc:
        return {"error": f"API request failed: {exc}"}
