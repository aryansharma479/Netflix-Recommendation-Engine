from typing import Dict
import numpy as np
import pandas as pd


# ---------------------------------------------------------
#                USER PROFILE GENERATION
# ---------------------------------------------------------
def build_user_profile(full: pd.DataFrame, user_id: int) -> Dict:
    df_u = full[full["userId"] == user_id].copy()
    if df_u.empty:
        return {"error": f"No data for user {user_id}"}

    avg_rating = df_u["rating"].mean()
    rating_std = df_u["rating"].std()
    n_ratings = df_u.shape[0]

    if avg_rating >= 4.2:
        behaviour = "Very generous rater"
    elif avg_rating >= 3.6:
        behaviour = "Slightly generous"
    elif avg_rating >= 3.0:
        behaviour = "Balanced"
    else:
        behaviour = "Strict rater"

    if "release_year" in df_u.columns:
        recent_mask = df_u["release_year"] >= 2010
        recent_like_ratio = (
            (df_u.loc[recent_mask, "rating"] >= 4).mean()
            if recent_mask.any()
            else np.nan
        )
    else:
        recent_like_ratio = np.nan

    # Helper to extract top categories
    def _top(col, sep="|", n=5):
        vals = df_u[col].fillna("").astype(str).str.split(sep)
        vals = vals.explode()
        vals = vals[vals != ""]
        if vals.empty:
            return []
        return vals.value_counts().head(n).index.tolist()

    top_genres = _top("genres", "|", 5) if "genres" in df_u.columns else []
    top_actors = _top("actors", "|", 5) if "actors" in df_u.columns else []
    top_directors = _top("directors", "|", 5) if "directors" in df_u.columns else []

    profile = {
        "user_id": int(user_id),
        "avg_rating": float(avg_rating),
        "rating_std": float(0 if pd.isna(rating_std) else rating_std),
        "n_ratings": int(n_ratings),
        "behaviour": behaviour,
        "top_genres": top_genres,
        "top_actors": top_actors,
        "top_directors": top_directors,
        "recent_like_ratio": None if pd.isna(recent_like_ratio) else float(recent_like_ratio),
    }
    return profile


# ---------------------------------------------------------
#                  MAIN RECOMMENDER
# ---------------------------------------------------------
def recommend_for_user(
    user_id: int,
    model,
    user_features: pd.DataFrame,
    movie_features: pd.DataFrame,
    movie_emb_df: pd.DataFrame,
    full: pd.DataFrame,
    movies_full: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:

    if user_id not in user_features.index:
        return pd.DataFrame()

    # Reindex movies_full by movieId (IMPORTANT FIX)
    if movies_full.index.name != "movieId":
        movies_full = movies_full.set_index("movieId")

    # Movies already rated by user
    seen = set(full.loc[full["userId"] == user_id, "movieId"].unique())
    candidate_movie_ids = [m for m in movie_features.index if m not in seen]
    if not candidate_movie_ids:
        return pd.DataFrame()

    # Extract user & movie features
    uf = user_features.loc[[user_id]]
    mf = movie_features.loc[candidate_movie_ids]
    emb = movie_emb_df.loc[candidate_movie_ids]
    nlp_cols = [c for c in emb.columns if c.startswith("nlp_")]

    # Repeat user row for candidate movies
    uf_rep = pd.concat([uf] * len(candidate_movie_ids), ignore_index=True)
    uf_rep.index = candidate_movie_ids

    # ---- Compose candidate feature frame ----
    X_cand = pd.concat(
        [
            uf_rep[
                [
                    "user_avg_rating",
                    "user_rating_std",
                    "user_rating_count",
                    "user_like_ratio",
                    "user_new_release_ratio",
                    "user_strictness",
                ]
            ],
            mf[
                [
                    "movie_avg_rating",
                    "movie_rating_count",
                    "movie_popularity",
                    "movie_like_ratio",
                    "release_year",
                    "years_since_release",
                    "is_trending",
                ]
            ],
            emb[nlp_cols],
        ],
        axis=1,
    ).fillna(0.0)

    # -------------------------------------------------------------
    # FIX — Add user_fav_genre_match (Safe, with reindexing)
    # -------------------------------------------------------------
    fav_genre = None
    if "user_fav_genre" in user_features.columns:
        try:
            fav_genre = user_features.loc[user_id, "user_fav_genre"]
        except:
            fav_genre = None

    def genre_match(movie_id):
        if movie_id not in movies_full.index:
            return 0
        g = movies_full.loc[movie_id, "genres"]
        if isinstance(fav_genre, str) and isinstance(g, str):
            return int(fav_genre in g.split("|"))
        return 0

    X_cand["user_fav_genre_match"] = X_cand.index.map(genre_match)

    # -------------------------------------------------------------
    # Align feature order with training model's feature names
    # -------------------------------------------------------------
    if hasattr(model, "feature_names_in_"):
        X_cand = X_cand.reindex(columns=model.feature_names_in_, fill_value=0.0)

    # -------------------------------------------------------------
    # Predict like probability
    # -------------------------------------------------------------
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_cand)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_cand)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    else:
        proba = model.predict(X_cand).astype(float)

    # Sort & merge metadata
    recs = (
        pd.DataFrame({"movieId": candidate_movie_ids, "pred_like_proba": proba})
        .sort_values("pred_like_proba", ascending=False)
        .head(top_n)
        .merge(movies_full, on="movieId", how="left")
    )

    return recs


# ---------------------------------------------------------
#            CONTENT-BASED SIMILARITY SEARCH
# ---------------------------------------------------------
def get_similar_movies(
    movie_id: int,
    movie_emb_df: pd.DataFrame,
    movies_full: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:

    if movie_id not in movie_emb_df.index:
        return pd.DataFrame()

    # Similarity = cosine similarity
    target_vec = movie_emb_df.loc[movie_id].values
    mat = movie_emb_df.values

    norms = (np.linalg.norm(mat, axis=1) * np.linalg.norm(target_vec))
    norms = np.where(norms == 0, 1e-9, norms)
    sims = np.dot(mat, target_vec) / norms

    sim_df = pd.DataFrame(
        {"movieId": movie_emb_df.index.values, "similarity": sims}
    )

    sim_df = sim_df[sim_df["movieId"] != movie_id]
    sim_df = sim_df.sort_values("similarity", ascending=False).head(top_n)

    # Reindex movies_full safely
    if movies_full.index.name != "movieId":
        movies_full = movies_full.set_index("movieId")

    sim_df = sim_df.merge(movies_full, on="movieId", how="left")
    return sim_df


# ---------------------------------------------------------
#             PURE SVD MATRIX-FACTORIZATION RECS
# ---------------------------------------------------------
def recommend_for_user_svd(
    user_id: int,
    svd_U: np.ndarray,
    svd_sigma: np.ndarray,
    svd_Vt: np.ndarray,
    full: pd.DataFrame,
    movies_full: pd.DataFrame,
    top_n: int = 10,
):
    """
    Pure SVD Recommender: R ≈ U * Σ * Vt
    """
    # Build user/movie id mapping
    user_ids = sorted(full["userId"].unique())
    movie_ids = sorted(full["movieId"].unique())

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    if user_id not in user_to_idx:
        return pd.DataFrame()

    # Full predicted rating matrix
    R_pred = np.dot(np.dot(svd_U, np.diag(svd_sigma)), svd_Vt)

    user_idx = user_to_idx[user_id]
    user_scores = R_pred[user_idx]

    seen_movies = set(full.loc[full["userId"] == user_id, "movieId"].unique())

    # Build candidate list
    preds = []
    for mid in movie_ids:
        if mid not in seen_movies:
            m_idx = movie_to_idx[mid]
            preds.append((mid, user_scores[m_idx]))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]

    rec_df = pd.DataFrame(preds, columns=["movieId", "svd_pred_rating"])

    # Reindex metadata safely
    if movies_full.index.name != "movieId":
        movies_full = movies_full.set_index("movieId")

    rec_df = rec_df.merge(movies_full, on="movieId", how="left")
    return rec_df
