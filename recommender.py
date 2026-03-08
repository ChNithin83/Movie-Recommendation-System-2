"""
recommender.py
──────────────
Core ML recommendation engine.

Pipeline:
  1. Load movies_data into a DataFrame
  2. Build TF-IDF feature matrix  (genres + cast + director + keywords)
  3. Compute full Cosine Similarity matrix
  4. Fit KNN NearestNeighbors model (brute, cosine metric)
  5. Expose three public methods:
       get_recommendations_cosine(title, n)
       get_recommendations_knn(title, n)
       get_combined_recommendations(title, n)
  6. Optionally enrich results with TMDB live metadata
"""

import numpy as np
import pandas as pd
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on="title")
import ast 
movies["genres"] = movies["genres"].astype(str)
movies["cast"] = movies["cast"].astype(str)
movies["keywords"] = movies["keywords"].astype(str)

def get_director(crew):
    for person in ast.literal_eval(crew):
        if person["job"] == "Director":
            return person["name"]
    return ""   

movies["director"] = movies["crew"].apply(get_director)
MOVIES = movies.to_dict(orient="records")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


import tmdb_client

# ── Build the data layer ───────────────────────────────────────────────────────

df = pd.DataFrame(MOVIES)
df.index = range(len(df))
df["tags"] = (
    df["title"] + " " +
    df["director"] + " " +
    df["cast"] + " " +
    df["genres"]
)
print(df.columns)
# ── TF-IDF Vectorization ───────────────────────────────────────────────────────

tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),      # unigrams + bigrams
    max_features=5000
)
# Combine important columns
df["features"] = (
    df["genres"].fillna("") + " " +
    df["keywords"].fillna("") + " " +
    df["overview"].fillna("") + " " +
    df["cast"].fillna("") + " " +
    df["director"].fillna("")
)
tfidf_matrix = tfidf.fit_transform(df["features"])

# ── Cosine Similarity Matrix ───────────────────────────────────────────────────

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ── KNN Model ─────────────────────────────────────────────────────────────────

knn_model = NearestNeighbors(
    n_neighbors=11,          # +1 because the movie itself is included
    algorithm="brute",
    metric="cosine"
)
knn_model.fit(tfidf_matrix)

# ── Utility Helpers ───────────────────────────────────────────────────────────

def _get_movie_index(title: str) -> int | None:
    """Return DataFrame index for a movie title (case-insensitive)."""
    title_lower = title.strip().lower()
    matches = df[df["title"].str.lower() == title_lower]
    if matches.empty:
        # Fuzzy partial match
        matches = df[df["title"].str.lower().str.contains(title_lower, na=False)]
    if matches.empty:
        return None
    return matches.index[0]


def _enrich(rec: dict) -> dict:
    """Add TMDB live data to a recommendation dict."""
    if not tmdb_client.is_api_available():
        return rec
    info = tmdb_client.fetch_movie_info(rec["title"], rec.get("id"))
    rec.update(info)
    return rec


def _build_rec(idx: int, score: float) -> dict:
    """Build a base recommendation dict from a DataFrame row."""
    row = df.iloc[idx]
    return {
        "id":           int(row.get("id", 0)),
        "title":        row["title"],
        "genres":       row["genres"],
        "director":     row["director"],
        "vote_average": float(row.get("vote_average", 0)),
        "release_year": str(int(row.get("release_year", 0))),
        "overview":     row.get("overview", ""),
        "similarity":   round(float(score), 4),
        # TMDB fields (populated by _enrich if API key available)
        "poster_url":   tmdb_client.FALLBACK_POSTER,
        "runtime":      0,
        "tagline":      "",
    }


# ── Public Recommendation Methods ─────────────────────────────────────────────

def get_recommendations_cosine(title: str, n: int = 5) -> list[dict]:
    """
    Content-based recommendations using Cosine Similarity.
    Returns top-n most similar movies (excluding the query movie).
    """
    idx = _get_movie_index(title)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]

    results = []
    for movie_idx, score in sim_scores:
        rec = _build_rec(movie_idx, score)
        rec = _enrich(rec)
        results.append(rec)
    return results


def get_recommendations_knn(title: str, n: int = 5) -> list[dict]:
    """
    Content-based recommendations using K-Nearest Neighbors.
    Returns top-n nearest neighbours (cosine distance, brute-force).
    """
    idx = _get_movie_index(title)
    if idx is None:
        return []

    movie_vector = tfidf_matrix[idx]
    distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=min(n + 1, len(df)))

    results = []
    for dist, movie_idx in zip(distances[0], indices[0]):
        if movie_idx == idx:
            continue
        similarity = 1 - dist          # convert cosine distance → similarity
        rec = _build_rec(movie_idx, similarity)
        rec = _enrich(rec)
        results.append(rec)
        if len(results) == n:
            break
    return results


def get_combined_recommendations(title: str, n: int = 5) -> list[dict]:
    """
    Hybrid engine: averages Cosine Similarity score and KNN score.
    Produces better precision than either method alone.
    """
    idx = _get_movie_index(title)
    if idx is None:
        return []

    # Cosine scores
    cosine_scores = {i: s for i, s in enumerate(cosine_sim[idx]) if i != idx}

    # KNN scores
    movie_vector = tfidf_matrix[idx]
    distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=len(df))
    knn_scores = {int(i): 1 - float(d) for d, i in zip(distances[0], indices[0]) if i != idx}

    # Combine: weighted average (0.5 + 0.5)
    all_indices = set(cosine_scores) | set(knn_scores)
    combined = {}
    for i in all_indices:
        c = cosine_scores.get(i, 0)
        k = knn_scores.get(i, 0)
        combined[i] = (c + k) / 2

    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n]
    results = []
    for movie_idx, score in top:
        rec = _build_rec(movie_idx, score)
        rec = _enrich(rec)
        results.append(rec)
    return results


# ── Dataset Info ──────────────────────────────────────────────────────────────

def get_all_titles() -> list[str]:
    return df["title"].tolist()


def get_genre_distribution() -> dict:
    genre_counts: dict[str, int] = {}
    for genres_str in df["genres"]:
        for g in genres_str.split():
            if len(g) > 3:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    return dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15])


def compute_similarity_score(title_a: str, title_b: str) -> float:
    """Return the cosine similarity score between two movie titles."""
    ia = _get_movie_index(title_a)
    ib = _get_movie_index(title_b)
    if ia is None or ib is None:
        return 0.0
    return round(float(cosine_sim[ia][ib]), 4)


def get_movie_info(title: str) -> dict | None:
    """Return raw row data for a movie title."""
    idx = _get_movie_index(title)
    if idx is None:
        return None
    return df.iloc[idx].to_dict()
