from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .config import RANDOM_STATE

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# -------------------------------------------------------------------
# DATA SPLITS (same as before)
# -------------------------------------------------------------------
def _ensure_no_cold_start_users(
    X_train, X_test, y_train, y_test, meta_train, meta_test
):
    train_users = set(meta_train["userId"].unique())
    test_users = set(meta_test["userId"].unique())

    cold_users = test_users - train_users
    if not cold_users:
        return X_train, X_test, y_train, y_test, meta_train, meta_test

    mask_cold = meta_test["userId"].isin(cold_users)
    X_move = X_test[mask_cold]
    y_move = y_test[mask_cold]
    meta_move = meta_test[mask_cold]

    X_test = X_test[~mask_cold]
    y_test = y_test[~mask_cold]
    meta_test = meta_test[~mask_cold]

    X_train = pd.concat([X_train, X_move], axis=0)
    y_train = pd.concat([y_train, y_move], axis=0)
    meta_train = pd.concat([meta_train, meta_move], axis=0)

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Dict[str, pd.DataFrame]:

    X_train_val, X_test, y_train_val, y_test, meta_train_val, meta_test = train_test_split(
        X, y, meta,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_train_val, y_train_val, meta_train_val,
        test_size=val_fraction,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test, meta_train, meta_test = _ensure_no_cold_start_users(
        X_train, X_test, y_train, y_test, meta_train, meta_test
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
    }


# -------------------------------------------------------------------
# BASELINE MODELS (same as before)
# -------------------------------------------------------------------
def build_model_pipelines() -> Dict[str, object]:
    models: Dict[str, object] = {}

    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    models["logistic_regression"] = logreg

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    models["random_forest"] = rf

    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        models["xgboost"] = xgb
    else:
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        )
        models["gradient_boosting"] = gb

    return models


def _metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = np.nan
        out["roc_auc"] = auc
    return out


def train_models(
    splits: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, object], pd.DataFrame]:

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    models = build_model_pipelines()
    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        # VAL
        y_val_pred = model.predict(X_val)
        y_val_proba = (
            model.predict_proba(X_val)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        m_val = _metrics(y_val, y_val_pred, y_val_proba)
        m_val.update({"model": name, "split": "val", "search": "baseline"})
        results.append(m_val)

        # TEST
        y_test_pred = model.predict(X_test)
        y_test_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
        m_test = _metrics(y_test, y_test_pred, y_test_proba)
        m_test.update({"model": name, "split": "test", "search": "baseline"})
        results.append(m_test)

    metrics_df = pd.DataFrame(results)
    return fitted, metrics_df


# -------------------------------------------------------------------
# HYPERPARAMETER TUNING (GridSearchCV / RandomizedSearchCV)
# -------------------------------------------------------------------
def tune_and_evaluate_models(
    splits: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, object], pd.DataFrame]:

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    tuned: Dict[str, object] = {}
    results = []

    # 1) Logistic Regression – GridSearchCV
    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    param_grid_lr = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "liblinear"],
    }
    grid_lr = GridSearchCV(
        logreg,
        param_grid_lr,
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    tuned_name_lr = "logistic_regression_tuned"
    tuned[tuned_name_lr] = best_lr

    for split_name, X_split, y_split in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = best_lr.predict(X_split)
        y_proba = best_lr.predict_proba(X_split)[:, 1]
        m = _metrics(y_split, y_pred, y_proba)
        m.update(
            {
                "model": tuned_name_lr,
                "split": split_name,
                "search": "grid",
                "best_params": str(grid_lr.best_params_),
            }
        )
        results.append(m)

    # 2) Random Forest – RandomizedSearchCV
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    param_dist_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    rand_rf = RandomizedSearchCV(
        rf,
        param_distributions=param_dist_rf,
        n_iter=20,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rand_rf.fit(X_train, y_train)
    best_rf = rand_rf.best_estimator_
    tuned_name_rf = "random_forest_tuned"
    tuned[tuned_name_rf] = best_rf

    for split_name, X_split, y_split in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = best_rf.predict(X_split)
        y_proba = best_rf.predict_proba(X_split)[:, 1]
        m = _metrics(y_split, y_pred, y_proba)
        m.update(
            {
                "model": tuned_name_rf,
                "split": split_name,
                "search": "random",
                "best_params": str(rand_rf.best_params_),
            }
        )
        results.append(m)

    # 3) Gradient Boosting OR XGBoost – small RandomizedSearch
    if XGBClassifier is not None:
        base_boost = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        param_dist_boost = {
            "n_estimators": [200, 400],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        }
    else:
        base_boost = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_dist_boost = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [2, 3],
        }

    rand_boost = RandomizedSearchCV(
        base_boost,
        param_distributions=param_dist_boost,
        n_iter=15,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rand_boost.fit(X_train, y_train)
    best_boost = rand_boost.best_estimator_
    tuned_name_boost = (
        "xgboost_tuned" if XGBClassifier is not None else "gradient_boosting_tuned"
    )
    tuned[tuned_name_boost] = best_boost

    for split_name, X_split, y_split in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = best_boost.predict(X_split)
        if hasattr(best_boost, "predict_proba"):
            y_proba = best_boost.predict_proba(X_split)[:, 1]
        else:
            scores = best_boost.decision_function(X_split)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        m = _metrics(y_split, y_pred, y_proba)
        m.update(
            {
                "model": tuned_name_boost,
                "split": split_name,
                "search": "random",
                "best_params": str(rand_boost.best_params_),
            }
        )
        results.append(m)

    metrics_df = pd.DataFrame(results)
    return tuned, metrics_df


# -------------------------------------------------------------------
# PURE NUMPY SVD FOR COLLABORATIVE FILTERING
# -------------------------------------------------------------------
def build_svd_factors(
    ratings: pd.DataFrame,
    k: int = 50,
):
    """
    Build user and item latent factors using plain NumPy SVD.

    - ratings: columns [userId, movieId, rating]
    - k: number of latent factors

    Returns:
        user_factors_df: index = userId, columns = svd_u_0..k-1
        item_factors_df: index = movieId, columns = svd_i_0..k-1
        user_bias:      Series of per-user mean rating
    """
    # user-item utility matrix
    R = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0.0)
    user_ids = R.index.values
    movie_ids = R.columns.values

    R_mat = R.values
    # user mean-centering
    user_mean = R_mat.mean(axis=1, keepdims=True)
    R_centered = R_mat - user_mean

    # SVD
    U, s, Vt = np.linalg.svd(R_centered, full_matrices=False)
    k = min(k, len(s))
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    # distribute singular values
    sqrt_s = np.sqrt(s_k)
    user_factors = U_k * sqrt_s
    item_factors = (Vt_k.T * sqrt_s)

    user_factors_df = pd.DataFrame(
        user_factors,
        index=user_ids,
        columns=[f"svd_u_{i}" for i in range(k)],
    )
    item_factors_df = pd.DataFrame(
        item_factors,
        index=movie_ids,
        columns=[f"svd_i_{i}" for i in range(k)],
    )

    user_bias = pd.Series(
        user_mean.squeeze(),
        index=user_ids,
        name="user_bias",
    )

    return user_factors_df, item_factors_df, user_bias
