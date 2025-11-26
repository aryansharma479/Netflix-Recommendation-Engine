import streamlit as st
import pandas as pd

from src.data_loader import load_raw_data, merge_movies_and_metadata, build_full_ratings_table
from src.features import (
    clean_full_table,
    build_text_embeddings,
    compute_user_features,
    compute_movie_features,
    build_interaction_dataset,
    build_movie_embedding_matrix,
)
from src.models import (
    train_val_test_split,
    train_models,
    tune_and_evaluate_models,
    build_svd_factors,
)
from src.recommender import (
    build_user_profile,
    recommend_for_user,
    recommend_for_user_svd,
    get_similar_movies,
)


st.set_page_config(
    page_title="Netflix Recommendation Engine",
    layout="wide",
)


@st.cache_data(show_spinner=True)
def load_and_prepare():
    ratings, movies, meta, genome_tags, genome_scores = load_raw_data()
    movies_full = merge_movies_and_metadata(movies, meta)
    full = build_full_ratings_table(ratings, movies_full)
    full = clean_full_table(full)

    text_emb_df, tfidf, svd_text = build_text_embeddings(movies_full)

    user_features = compute_user_features(full)
    movie_features = compute_movie_features(full)
    movie_emb_df = build_movie_embedding_matrix(movie_features, text_emb_df)

    X, y, meta_X, enriched_df = build_interaction_dataset(
        full, user_features, movie_features, text_emb_df
    )

    # NEW: build SVD latent factors for collaborative filtering
    svd_user_factors, svd_movie_factors, svd_user_bias = build_svd_factors(ratings)

    return {
        "ratings": ratings,
        "movies": movies,
        "movies_full": movies_full,
        "full": enriched_df,
        "user_features": user_features,
        "movie_features": movie_features,
        "movie_emb_df": movie_emb_df,
        "X": X,
        "y": y,
        "meta_X": meta_X,
        "svd_user_factors": svd_user_factors,
        "svd_movie_factors": svd_movie_factors,
        "svd_user_bias": svd_user_bias,
    }


@st.cache_resource(show_spinner=True)
def train_all_models(X, y, meta_X):
    splits = train_val_test_split(X, y, meta_X)

    # Baseline models
    base_models, base_metrics_df = train_models(splits)

    # Hyperparameter-tuned models (GridSearchCV / RandomizedSearchCV)
    tuned_models, tuned_metrics_df = tune_and_evaluate_models(splits)

    # Merge models and metrics
    all_models = {**base_models, **tuned_models}
    all_metrics = pd.concat([base_metrics_df, tuned_metrics_df], ignore_index=True)

    return all_models, all_metrics, splits


def main():
    st.title("🎬 Netflix Recommendation Engine")
    st.markdown(
        """
        Builds an end-to-end ML system with:
        - Data cleaning & feature engineering  
        - User profiles & rating behaviour  
        - NLP-driven content embeddings (TF-IDF + SVD)  
        - Multiple models (LogReg, RF, XGBoost/GBM)  
        - Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
        - Collaborative Filtering (pure NumPy SVD)  
        - Personalized recommendations & similar content
        """
    )

    data = load_and_prepare()

    full = data["full"]
    movies_full = data["movies_full"]
    user_features = data["user_features"]
    movie_features = data["movie_features"]
    movie_emb_df = data["movie_emb_df"]
    X, y, meta_X = data["X"], data["y"], data["meta_X"]

    svd_user_factors = data["svd_user_factors"]
    svd_movie_factors = data["svd_movie_factors"]
    svd_user_bias = data["svd_user_bias"]

    models, metrics_df, splits = train_all_models(X, y, meta_X)

    tab_eda, tab_models, tab_user, tab_similar = st.tabs(
        ["📊 Data & EDA", "🤖 Models", "👤 Recommendations", "🎞 Similar Content"]
    )

    # ------------------ TAB 1: EDA ------------------
    with tab_eda:
        st.subheader("Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Users", full["userId"].nunique())
        c2.metric("Movies", full["movieId"].nunique())
        c3.metric("Ratings", len(full))
        c4.metric("Avg Rating", round(full["rating"].mean(), 2))

        st.markdown("### Sample Interactions")
        st.dataframe(full.head())

        st.markdown("### Rating Distribution")
        if "rating" not in full.columns:
            st.error("'rating' column missing.")
            st.write("Columns:", list(full.columns))
        else:
            counts = full["rating"].value_counts().sort_index()
            rating_counts = pd.DataFrame(
                {"rating": counts.index.astype(float), "count": counts.values}
            )
            st.bar_chart(rating_counts.set_index("rating"))

        st.markdown("### Top Genres")
        if "genres" in full.columns:
            tmp = full["genres"].fillna("").str.split("|").explode()
            tmp = tmp[tmp != ""]
            genre_counts = tmp.value_counts().head(15).to_frame("count")
            st.bar_chart(genre_counts)
        else:
            st.info("No 'genres' column available.")

    # ------------------ TAB 2: MODELS ------------------
    with tab_models:
        st.subheader("Model Performance (Baseline & Tuned)")

        if metrics_df.empty:
            st.error("No metrics computed.")
        else:
            st.markdown("### Summary Metrics")
            st.dataframe(
                metrics_df.pivot_table(
                    index="model",
                    columns=["split", "search"],
                    values=["accuracy", "precision", "recall", "f1", "roc_auc"],
                ).round(3)
            )

            st.markdown("#### Test F1 & ROC-AUC (All Models)")
            test_metrics = metrics_df[metrics_df["split"] == "test"].set_index("model")
            chart_df = test_metrics[["f1", "roc_auc"]]
            st.bar_chart(chart_df)

            st.info(
                """
                - **Baseline models** use default hyperparameters.  
                - **Tuned models** use GridSearchCV (LogReg) and RandomizedSearchCV (RF / GB/XGBoost).  
                - Compare baseline vs tuned F1 / ROC-AUC to see the impact of hyperparameter tuning.
                """
            )

            st.markdown("### Best Parameters (Tuned Models)")

            # Keep only rows where best_params exists
            if "best_params" in metrics_df.columns:
                tuned_only = (
                    metrics_df[
                        (metrics_df["search"] != "baseline")
                        & (metrics_df["split"] == "val")
                        & (metrics_df["best_params"].notna())
                    ][["model", "best_params"]]
                    .drop_duplicates()
                    .set_index("model")
                )

                if not tuned_only.empty:
                    st.dataframe(tuned_only)
                else:
                    st.info("No tuned model parameter details found.")
            else:
                st.info("Hyperparameter tuning not available.")

    # ------------------ TAB 3: USER PROFILES & RECS ------------------
    with tab_user:
        st.subheader("User Profiles & Personalized Recommendations")

        user_ids = sorted(full["userId"].unique())
        user_id = st.selectbox("Choose a user", user_ids)

        profile = build_user_profile(full, user_id)
        if "error" in profile:
            st.warning(profile["error"])
        else:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("#### Rating Behaviour")
                st.write(f"User ID: **{profile['user_id']}**")
                st.write(f"Average rating: **{profile['avg_rating']:.2f}**")
                st.write(f"Total ratings: **{profile['n_ratings']}**")
                st.write(f"Behaviour: **{profile['behaviour']}**")
                if profile["recent_like_ratio"] is not None:
                    st.write(
                        f"Likes recent releases: **{profile['recent_like_ratio']:.2f}** "
                        "(fraction of recent movies rated ≥ 4)"
                    )

            with col_r:
                st.markdown("#### Content Preferences")
                st.write("Top genres:", ", ".join(profile["top_genres"]) or "N/A")
                st.write("Top actors:", ", ".join(profile["top_actors"]) or "N/A")
                st.write("Top directors:", ", ".join(profile["top_directors"]) or "N/A")

        st.markdown("---")
        st.markdown("### ML Model-based Recommendations")

        model_name = st.selectbox("Model", list(models.keys()))
        top_n = st.slider("Number of recommendations", 5, 30, 10, key="ml_topn")

        if st.button("Get ML Recommendations"):
            model = models[model_name]
            recs = recommend_for_user(
                user_id=user_id,
                model=model,
                user_features=user_features,
                movie_features=movie_features,
                movie_emb_df=movie_emb_df,
                full=full,
                movies_full=movies_full,
                top_n=top_n,
            )
            if recs.empty:
                st.warning("No recommendations could be generated for this user.")
            else:
                for _, row in recs.iterrows():
                    with st.container():
                        c_img, c_txt = st.columns([1, 3])
                        with c_img:
                            if isinstance(row.get("poster_url"), str) and row["poster_url"]:
                                st.image(row["poster_url"], use_container_width=True)
                        with c_txt:
                            st.markdown(f"**{row['title']}**")
                            st.write(row.get("genres", ""))
                            st.write(
                                f"Predicted like probability: **{row['pred_like_proba']:.3f}**"
                            )

        st.markdown("---")
        st.markdown("### Collaborative Filtering (SVD-based) Recommendations")

        top_n_svd = st.slider(
            "Number of SVD CF recommendations", 5, 30, 10, key="svd_topn"
        )

        if st.button("Get SVD CF Recommendations"):
            recs_svd = recommend_for_user_svd(
                user_id=user_id,
                svd_user_factors=svd_user_factors,
                svd_movie_factors=svd_movie_factors,
                svd_user_bias=svd_user_bias,
                full=full,
                movies_full=movies_full,
                top_n=top_n_svd,
            )
            if recs_svd.empty:
                st.warning("No SVD CF recommendations could be generated for this user.")
            else:
                for _, row in recs_svd.iterrows():
                    with st.container():
                        c_img, c_txt = st.columns([1, 3])
                        with c_img:
                            if isinstance(row.get("poster_url"), str) and row["poster_url"]:
                                st.image(row["poster_url"], use_container_width=True)
                        with c_txt:
                            st.markdown(f"**{row['title']}**")
                            st.write(row.get("genres", ""))
                            st.write(
                                f"SVD score (higher = more preferred): **{row['svd_score']:.3f}**"
                            )

    # ------------------ TAB 4: SIMILAR CONTENT ------------------
    with tab_similar:
        st.subheader("Content-Based Similar Movies")

        movie_choices = movies_full[["movieId", "title"]].drop_duplicates()
        movie_choices["label"] = (
            movie_choices["title"] + " (ID: " + movie_choices["movieId"].astype(str) + ")"
        )
        label = st.selectbox(
            "Reference movie", movie_choices["label"].tolist()
        )
        movie_id = int(
            movie_choices.loc[movie_choices["label"] == label, "movieId"].iloc[0]
        )

        similar_df = get_similar_movies(
            movie_id=movie_id,
            movie_emb_df=movie_emb_df,
            movies_full=movies_full,
            top_n=15,
        )

        if similar_df.empty:
            st.warning("No similar movies found.")
        else:
            for _, row in similar_df.iterrows():
                with st.container():
                    c_img, c_txt = st.columns([1, 3])
                    with c_img:
                        if isinstance(row.get("poster_url"), str) and row["poster_url"]:
                            st.image(row["poster_url"], use_container_width=True)
                    with c_txt:
                        st.markdown(f"**{row['title']}**")
                        st.write(row.get("genres", ""))
                        st.write(f"Similarity score: **{row['similarity']:.3f}**")

            st.info(
                """
                Similarity is computed using:
                - TF-IDF + SVD embeddings of the **overview** text  
                - Combined with numerical movie features (ratings, popularity, recency)  
                using cosine similarity.
                """
            )


if __name__ == "__main__":
    main()
