"""Tests for tools/open_food_facts.py.

All tests mock requests.get — no real HTTP calls are made.
Covers: barcode lookup, name search, not-found, timeout, HTTP errors,
missing fields, allergen tag stripping, and URL routing.
"""

from __future__ import annotations

import pytest
import requests

from tools.open_food_facts import lookup_product, _is_barcode, _best_match, _OFF_BARCODE_URL, _OFF_SEARCH_URL

# ---------------------------------------------------------------------------
# Shared mock product data
# ---------------------------------------------------------------------------

MOCK_PRODUCT: dict = {
    "product_name": "Test Product",
    "ingredients_text": "sugar, palm oil, cocoa",
    "allergens_tags": ["en:milk", "en:eggs"],
    "nutriments": {
        "energy-kcal_100g": 539.0,
        "fat_100g": 30.9,
        "sugars_100g": 56.3,
        "proteins_100g": 6.3,
        "salt_100g": 0.107,
    },
    "nutriscore_grade": "e",
    "nova_group": 4,
}


def _make_mock_response(mocker, json_data: dict, status_code: int = 200):
    """Helper: build a mock requests.Response."""
    mock_resp = mocker.MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    return mock_resp


# ---------------------------------------------------------------------------
# Barcode detection helper
# ---------------------------------------------------------------------------


def test_is_barcode_digits():
    assert _is_barcode("3017620422003") is True


def test_is_barcode_name():
    assert _is_barcode("coca cola") is False


def test_is_barcode_mixed():
    assert _is_barcode("abc123") is False


# ---------------------------------------------------------------------------
# Best-match scoring
# ---------------------------------------------------------------------------


def test_best_match_picks_matching_product():
    """When multiple products returned, _best_match picks the one whose name matches."""
    products = [
        {"product_name": "Fromage Blanc Nature"},
        {"product_name": "Coca Cola Original Taste"},
        {"product_name": "Sidi Ali"},
    ]
    assert _best_match(products, "coca cola")["product_name"] == "Coca Cola Original Taste"


def test_best_match_falls_back_to_first():
    """When no product name matches any query word, return the first result."""
    products = [
        {"product_name": "Product A"},
        {"product_name": "Product B"},
    ]
    assert _best_match(products, "xyz")["product_name"] == "Product A"


def test_best_match_uses_product_name_en_fallback():
    """Scoring should fall back to product_name_en when product_name is empty."""
    products = [
        {"product_name": "", "product_name_en": "Nutella Hazelnut Spread"},
        {"product_name": "Random Product"},
    ]
    assert _best_match(products, "nutella")["product_name_en"] == "Nutella Hazelnut Spread"


# ---------------------------------------------------------------------------
# Happy path — barcode lookup
# ---------------------------------------------------------------------------


def test_barcode_happy_path(mocker):
    """Valid barcode → structured product dict with all expected keys."""
    mock_resp = _make_mock_response(
        mocker, {"status": 1, "product": MOCK_PRODUCT}
    )
    mocker.patch("tools.open_food_facts.requests.get", return_value=mock_resp)

    result = lookup_product.invoke({"query": "3017620422003"})

    assert result["product_name"] == "Test Product"
    assert result["allergens"] == ["milk", "eggs"]
    assert result["nutriments"]["calories_per_100g"] == 539.0
    assert result["nutriscore_grade"] == "e"
    assert result["nova_group"] == 4
    assert "error" not in result


def test_barcode_routes_to_correct_url(mocker):
    """Barcode query must use the barcode API endpoint."""
    mock_resp = _make_mock_response(
        mocker, {"status": 1, "product": MOCK_PRODUCT}
    )
    mock_get = mocker.patch("tools.open_food_facts.requests.get", return_value=mock_resp)

    lookup_product.invoke({"query": "3017620422003"})

    called_url = mock_get.call_args[0][0]
    assert "api/v2/product" in called_url
    assert "3017620422003" in called_url


# ---------------------------------------------------------------------------
# Happy path — name search
# ---------------------------------------------------------------------------


def test_name_search_happy_path(mocker):
    """Product name query → first result returned as structured dict."""
    mock_resp = _make_mock_response(
        mocker, {"products": [MOCK_PRODUCT]}
    )
    mocker.patch("tools.open_food_facts.requests.get", return_value=mock_resp)

    result = lookup_product.invoke({"query": "coca cola"})

    assert result["product_name"] == "Test Product"
    assert "error" not in result


def test_name_search_routes_to_search_url(mocker):
    """Name query must use the search endpoint, not the product endpoint."""
    mock_resp = _make_mock_response(
        mocker, {"products": [MOCK_PRODUCT]}
    )
    mock_get = mocker.patch("tools.open_food_facts.requests.get", return_value=mock_resp)

    lookup_product.invoke({"query": "coca cola"})

    called_url = mock_get.call_args[0][0]
    assert "search.pl" in called_url or "search_terms" in called_url


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_barcode_not_found(mocker):
    """API status=0 → error dict with product-not-found message."""
    mock_resp = _make_mock_response(mocker, {"status": 0})
    mocker.patch("tools.open_food_facts.requests.get", return_value=mock_resp)

    result = lookup_product.invoke({"query": "0000000000000"})

    assert "error" in result
    assert "not found" in result["error"].lower()


def test_name_search_empty_results(mocker):
    """Empty products list → error dict."""
    mock_resp = _make_mock_response(mocker, {"products": []})
    mocker.patch("tools.open_food_facts.requests.get", return_value=mock_resp)

    result = lookup_product.invoke({"query": "xyznonexistentproduct"})

    assert "error" in result
    assert "found" in result["error"].lower()


def test_timeout(mocker):
    """requests.Timeout → error dict with timeout message."""
    mocker.patch(
        "tools.open_food_facts.requests.get",
        side_effect=requests.exceptions.Timeout,
    )

    result = lookup_product.invoke({"query": "3017620422003"})

    assert "error" in result
    assert "timed out" in result["error"].lower()


def test_http_500_error(mocker):
    """HTTP 500 response → error dict with status code."""
    mock_resp = _make_mock_response(mocker, {}, status_code=500)
    mocker.patch("tools.open_food_facts.requests.get", return_value=mock_resp)

    result = lookup_product.invoke({"query": "3017620422003"})

    assert "error" in result
    assert "500" in result["error"]


def test_request_exception(mocker):
    """Generic RequestException → error dict with failure message."""
    mocker.patch(
        "tools.open_food_facts.requests.get",
        side_effect=requests.exceptions.RequestException("connection refused"),
    )

    result = lookup_product.invoke({"query": "3017620422003"})

    assert "error" in result
    assert "failed" in result["error"].lower()


# ---------------------------------------------------------------------------
# Field normalisation
# ---------------------------------------------------------------------------


def test_missing_nutriments_returns_none_values(mocker):
    """Product dict without 'nutriments' key → all nutriment values are None."""
    product_no_nutriments = {**MOCK_PRODUCT}
    del product_no_nutriments["nutriments"]

    mock_resp = _make_mock_response(
        mocker, {"status": 1, "product": product_no_nutriments}
    )
    mocker.patch("tools.open_food_facts.requests.get", return_value=mock_resp)

    result = lookup_product.invoke({"query": "3017620422003"})

    assert result["nutriments"]["calories_per_100g"] == "not available"
    assert result["nutriments"]["fat_per_100g"] == "not available"


def test_allergen_tags_stripped_of_language_prefix(mocker):
    """'en:milk' and 'en:peanuts' must be returned as 'milk' and 'peanuts'."""
    product = {
        **MOCK_PRODUCT,
        "allergens_tags": ["en:milk", "en:peanuts", "fr:gluten"],
    }
    mock_resp = _make_mock_response(
        mocker, {"status": 1, "product": product}
    )
    mocker.patch("tools.open_food_facts.requests.get", return_value=mock_resp)

    result = lookup_product.invoke({"query": "3017620422003"})

    assert "milk" in result["allergens"]
    assert "peanuts" in result["allergens"]
    assert "gluten" in result["allergens"]
    # No raw "en:..." prefixes in the result
    assert not any(a.startswith("en:") for a in result["allergens"])
