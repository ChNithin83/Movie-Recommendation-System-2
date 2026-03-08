"""
tmdb_client.py
──────────────
Handles all communication with The Movie Database (TMDB) REST API v3.
  - Fetches: poster_url, overview, release_year, vote_average, vote_count, runtime, tagline
  - Implements LRU caching (maxsize=256) to avoid duplicate API calls
  - Loads API key securely from .env file or Streamlit secrets
  - All calls wrapped in try/except with graceful fallback
"""

import os
import requests
from functools import lru_cache

# ── API Configuration ──────────────────────────────────────────────────────────
BASE_URL  = "https://api.tmdb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"
FALLBACK_POSTER = "https://via.placeholder.com/500x750/1a1a2e/ffffff?text=No+Poster"

def _get_api_key() -> str:
    """Load API key from Streamlit secrets (cloud) or .env file (local)."""
    # 1. Try Streamlit secrets (when deployed on Streamlit Cloud)
    try:
        import streamlit as st
        key = st.secrets.get("TMDB_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    # 2. Try .env file via python-dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # 3. Try environment variable directly
    return os.environ.get("TMDB_API_KEY", "")


API_KEY = _get_api_key()

# ── Core Fetch Helpers ─────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def search_movie(title: str) -> dict:
    """
    Search TMDB for a movie by title.
    Returns the first result dict or {} on failure.
    """
    if not API_KEY:
        return {}
    try:
        resp = requests.get(
            f"{BASE_URL}/search/movie",
            params={"api_key": API_KEY, "query": title, "language": "en-US"},
            timeout=8
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else {}
    except Exception:
        return {}


@lru_cache(maxsize=256)
def get_movie_details(tmdb_id: int) -> dict:
    """
    Fetch full details for a movie by its TMDB ID.
    Returns details dict or {} on failure.
    """
    if not API_KEY:
        return {}
    try:
        resp = requests.get(
            f"{BASE_URL}/movie/{tmdb_id}",
            params={"api_key": API_KEY, "language": "en-US"},
            timeout=8
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def get_poster_url(poster_path: str) -> str:
    """Convert a TMDB poster_path to a full image URL."""
    if poster_path:
        return f"{IMAGE_URL}{poster_path}"
    return FALLBACK_POSTER


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_movie_info(title: str, tmdb_id: int | None = None) -> dict:
    """
    Master function: fetch all enrichment data for a movie.
    Tries by TMDB ID first (faster), then falls back to title search.

    Returns a dict with keys:
        poster_url, overview, release_year, vote_average,
        vote_count, runtime, tagline, tmdb_id
    """
    details = {}

    # Try by known ID first
    if tmdb_id:
        details = get_movie_details(int(tmdb_id))

    # Fall back to search
    if not details:
        search = search_movie(title)
        if search:
            details = get_movie_details(search.get("id", 0))

    if not details:
        return _empty_info()

    release_year = ""
    rd = details.get("release_date", "")
    if rd and len(rd) >= 4:
        release_year = rd[:4]

    return {
        "poster_url":    get_poster_url(details.get("poster_path", "")),
        "overview":      details.get("overview", "No overview available."),
        "release_year":  release_year,
        "vote_average":  details.get("vote_average", 0.0),
        "vote_count":    details.get("vote_count", 0),
        "runtime":       details.get("runtime", 0),
        "tagline":       details.get("tagline", ""),
        "tmdb_id":       details.get("id", ""),
    }


def _empty_info() -> dict:
    return {
        "poster_url":   FALLBACK_POSTER,
        "overview":     "No overview available.",
        "release_year": "",
        "vote_average": 0.0,
        "vote_count":   0,
        "runtime":      0,
        "tagline":      "",
        "tmdb_id":      "",
    }


def is_api_available() -> bool:
    """Return True if a valid API key is configured."""
    return bool(API_KEY)
