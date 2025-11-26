from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .config import (
    LIKE_THRESHOLD,
    RANDOM_STATE,
    N_TEXT_COMPONENTS,
    NEW_RELEASE_YEAR,
    CURRENT_YEAR_FOR_RECENCY,
)


def clean_full_table(full: pd.DataFrame) -> pd.DataFrame:
    """Basic NA handling and type fixes on the interaction table."""
    df = full.copy()

    # Text fields
    for col in ["overview", "genres", "genres_meta", "actors", "directors"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Release year: numeric and fill from title if missing
    if "release_year" in df.columns:
        df["release_year"] = df["release_year"].fillna(0).astype(int)
        mask_zero = df["release_year"] == 0

        if "title" in df.columns:
            inferred = (
                df.loc[mask_zero, "title"].astype(str)
                .str.extract(r"\((\d{4})\)")
                .iloc[:, 0]
            )
            df.loc[mask_zero, "release_year"] = inferred.fillna(0).astype(int)

        median_year = df.loc[df["release_year"] > 0, "release_year"].median()
        df["release_year"] = df["release_year"].replace(0, median_year).astype(int)

    # rating
    if "rating" in df.columns:
        df["rating"] = df["rating"].astype(float)

    return df


def build_text_embeddings(movies_full: pd.DataFrame):
    """
    TF-IDF + SVD embeddings for movie overview text.
    """
    texts = movies_full["overview"].fillna("").astype(str).values

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = tfidf.fit_transform(texts)

    svd = TruncatedSVD(
        n_components=N_TEXT_COMPONENTS,
        random_state=RANDOM_STATE,
    )
    reduced = svd.fit_transform(tfidf_matrix)

    emb_df = pd.DataFrame(
        reduced,
        index=movies_full["movieId"].values,
        columns=[f"nlp_{i}" for i in range(N_TEXT_COMPONENTS)],
    )
    return emb_df, tfidf, svd


def compute_user_features(full: pd.DataFrame) -> pd.DataFrame:
    """User-level profile features."""
    df = full.copy()
    df["like"] = (df["rating"] >= LIKE_THRESHOLD).astype(int)

    stats = df.groupby("userId")["rating"].agg(["mean", "std", "count"])
    stats.rename(
        columns={
            "mean": "user_avg_rating",
            "std": "user_rating_std",
            "count": "user_rating_count",
        },
        inplace=True,
    )

    like_ratio = df.groupby("userId")["like"].mean().to_frame("user_like_ratio")

    if "release_year" in df.columns:
        df["is_new_release"] = (df["release_year"] >= NEW_RELEASE_YEAR).astype(int)
        new_release_pref = (
            df.groupby("userId")["is_new_release"]
            .mean()
            .to_frame("user_new_release_ratio")
        )
    else:
        new_release_pref = pd.DataFrame(
            {"user_new_release_ratio": 0.0}, index=stats.index
        )

    strictness = (5.0 - stats["user_avg_rating"]).to_frame("user_strictness")

    user_features = (
        stats.join(like_ratio)
        .join(new_release_pref, how="left")
        .join(strictness)
    )
    user_features["user_rating_std"] = user_features["user_rating_std"].fillna(0.0)

    return user_features


def compute_movie_features(full: pd.DataFrame) -> pd.DataFrame:
    """Movie-level stats & popularity."""
    df = full.copy()
    df["like"] = (df["rating"] >= LIKE_THRESHOLD).astype(int)

    stats = df.groupby("movieId")["rating"].agg(["mean", "count"])
    stats.rename(
        columns={
            "mean": "movie_avg_rating",
            "count": "movie_rating_count",
        },
        inplace=True,
    )

    like_ratio = df.groupby("movieId")["like"].mean().to_frame("movie_like_ratio")

    stats["movie_popularity"] = np.log1p(stats["movie_rating_count"])

    recency = df.groupby("movieId")["release_year"].median().to_frame("release_year")
    recency["years_since_release"] = CURRENT_YEAR_FOR_RECENCY - recency["release_year"]
    recency["years_since_release"] = recency["years_since_release"].clip(lower=0)
    recency["is_trending"] = (recency["release_year"] >= NEW_RELEASE_YEAR).astype(int)

    movie_features = stats.join(like_ratio).join(recency)
    return movie_features


def build_interaction_dataset(
    full: pd.DataFrame,
    user_features: pd.DataFrame,
    movie_features: pd.DataFrame,
    text_emb_df: pd.DataFrame,
):
    """
    Build per (user, movie) rows with engineered features & target.
    """
    df = full.copy()
    df["like"] = (df["rating"] >= LIKE_THRESHOLD).astype(int)

    # Avoid duplicate release_year when joining movie_features
    if "release_year" in df.columns:
        df.drop(columns=["release_year"], inplace=True)

    df = df.join(user_features, on="userId", how="left")
    df = df.join(movie_features, on="movieId", how="left")
    df = df.join(text_emb_df, on="movieId", how="left")

    # User favourite genre match
    if "genres" in df.columns:
        tmp = df[["userId", "genres"]].copy()
        tmp["genres_list"] = tmp["genres"].fillna("").str.split("|")
        tmp = tmp.explode("genres_list")
        tmp = tmp[tmp["genres_list"] != ""]
        fav_genre = (
            tmp.groupby(["userId", "genres_list"])
            .size()
            .reset_index(name="cnt")
            .sort_values(["userId", "cnt"], ascending=[True, False])
        )
        fav_genre = fav_genre.drop_duplicates("userId").set_index("userId")[
            "genres_list"
        ]
        df["user_fav_genre"] = df["userId"].map(fav_genre)
        df["user_fav_genre_match"] = df.apply(
            lambda row: int(
                isinstance(row["user_fav_genre"], str)
                and isinstance(row["genres"], str)
                and row["user_fav_genre"] in row["genres"].split("|")
            ),
            axis=1,
        )
    else:
        df["user_fav_genre_match"] = 0

    nlp_cols = [c for c in df.columns if c.startswith("nlp_")]

    feature_cols = [
        "user_avg_rating",
        "user_rating_std",
        "user_rating_count",
        "user_like_ratio",
        "user_new_release_ratio",
        "user_strictness",
        "movie_avg_rating",
        "movie_rating_count",
        "movie_popularity",
        "movie_like_ratio",
        "release_year",
        "years_since_release",
        "is_trending",
        "user_fav_genre_match",
    ] + nlp_cols

    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0.0)
    y = df["like"].astype(int)
    meta = df[["userId", "movieId"]].copy()

    return X, y, meta, df


def build_movie_embedding_matrix(
    movie_features: pd.DataFrame,
    text_emb_df: pd.DataFrame,
) -> pd.DataFrame:
    emb = movie_features.join(text_emb_df, how="left")
    emb = emb.fillna(0.0)
    return emb
