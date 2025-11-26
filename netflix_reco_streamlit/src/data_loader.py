import os
import pandas as pd
from .config import (
    DATA_DIR,
    RATINGS_FILE,
    MOVIES_FILE,
    META_FILE,
    GENOME_TAGS_FILE,
    GENOME_SCORES_FILE,
)


def load_raw_data(data_dir: str = DATA_DIR):
    """Load raw CSVs from the data directory."""
    ratings_path = os.path.join(data_dir, RATINGS_FILE)
    movies_path = os.path.join(data_dir, MOVIES_FILE)
    meta_path = os.path.join(data_dir, META_FILE)
    genome_tags_path = os.path.join(data_dir, GENOME_TAGS_FILE)
    genome_scores_path = os.path.join(data_dir, GENOME_SCORES_FILE)

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    meta = pd.read_csv(meta_path)
    genome_tags = pd.read_csv(genome_tags_path)
    genome_scores = pd.read_csv(genome_scores_path)

    if "rating" not in ratings.columns:
        raise ValueError(
            f"'rating' column missing in ratings.csv. "
            f"Columns found: {ratings.columns.tolist()}"
        )

    return ratings, movies, meta, genome_tags, genome_scores


def merge_movies_and_metadata(movies: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Merge MovieLens movies with metadata.
    We keep MovieLens title + genres as canonical, and rename
    metadata genres to genres_meta to avoid x/y suffixes.
    """
    meta_clean = meta.copy()

    # Avoid clashing with MovieLens columns
    if "title" in meta_clean.columns:
        meta_clean = meta_clean.drop(columns=["title"])

    if "genres" in meta_clean.columns:
        meta_clean = meta_clean.rename(columns={"genres": "genres_meta"})

    movies_full = movies.merge(meta_clean, on="movieId", how="left")
    # movies_full now has: movieId, title, genres, tmdb_id, overview, poster_url, genres_meta, actors, ...
    return movies_full


def build_full_ratings_table(
    ratings: pd.DataFrame, movies_full: pd.DataFrame
) -> pd.DataFrame:
    """
    Join ratings with movie+metadata table.
    """
    full = ratings.merge(movies_full, on="movieId", how="left")
    return full
