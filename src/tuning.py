"""Stages 7-12 - Baseline, ensembles, stacking, tuning and cross-validation.

Everything here uses ONLY the training split. Every model is scored with the
same StratifiedKFold(5, shuffle=True, random_state=42) and f1_macro, so CV
numbers are directly comparable across models.

For each model the script records: best parameters, mean / std CV Macro F1,
fold-wise scores, mean training score (to expose overfitting), fit time.
Each tuned pipeline is refitted on the full training split and saved to
models/candidates/ for the final evaluation in Stage 13.

    python -m src.tuning
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate

from config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.feature_study import load_selected_feature_params, load_split
from src.models import (
    CATBOOST_SPACE,
    LOGISTIC_GRID,
    RANDOM_FOREST_SPACE,
    STACKING_GRID,
    XGBOOST_SPACE,
    catboost_model,
    dummy_model,
    logistic_model,
    make_cv,
    make_pipeline,
    random_forest_model,
    stacking_model,
    strip_prefix,
    xgboost_model,
)
from src.utils import ensure_dirs

TUNING_DIR = REPORTS_DIR / "tuning"
CANDIDATES_DIR = MODELS_DIR / "candidates"
N_JOBS = 2
SCORING = "f1_macro"

MODEL_LABELS = {
    "dummy": "Baseline (Dummy, most frequent)",
    "logistic_regression": "Baseline (Logistic Regression)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "stacking": "Stacking",
}


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def cv_only(key: str, pipeline, X, y, note: str) -> tuple[dict, object]:
    """Plain 5-fold CV for a model with fixed parameters (no search)."""
    start = time.time()
    result = cross_validate(pipeline, X, y, cv=make_cv(), scoring=SCORING, return_train_score=True, n_jobs=N_JOBS,
                            error_score="raise")
    fitted = pipeline.fit(X, y)
    record = {
        "model": key,
        "label": MODEL_LABELS.get(key, key),
        "search": note,
        "n_candidates": 1,
        "best_params": {},
        "cv_macro_f1_mean": float(result["test_score"].mean()),
        "cv_macro_f1_std": float(result["test_score"].std()),
        "train_macro_f1_mean": float(result["train_score"].mean()),
        "fold_scores": [float(s) for s in result["test_score"]],
        "seconds": round(time.time() - start, 1),
    }
    return record, fitted


def search(key: str, pipeline, space: dict, X, y, kind: str, n_iter: int | None = None) -> tuple[dict, object]:
    """RandomizedSearchCV (large spaces) or GridSearchCV (small final grids)."""
    # error_score="raise": a failing candidate must stop the run, not be
    # silently scored as NaN and skipped.
    common = dict(scoring=SCORING, cv=make_cv(), n_jobs=N_JOBS, refit=True, return_train_score=True,
                  error_score="raise")
    if kind == "random":
        searcher = RandomizedSearchCV(pipeline, space, n_iter=n_iter, random_state=RANDOM_STATE, **common)
    else:
        searcher = GridSearchCV(pipeline, space, **common)

    start = time.time()
    searcher.fit(X, y)
    elapsed = time.time() - start

    results = pd.DataFrame(searcher.cv_results_)
    results.to_csv(TUNING_DIR / f"{key}_search_results.csv", index=False)
    best = searcher.best_index_
    folds = [float(results.loc[best, f"split{i}_test_score"]) for i in range(make_cv().get_n_splits())]
    record = {
        "model": key,
        "label": MODEL_LABELS[key],
        "search": "RandomizedSearchCV" if kind == "random" else "GridSearchCV",
        "n_candidates": int(len(results)),
        "best_params": {k: _jsonable(v) for k, v in searcher.best_params_.items()},
        "cv_macro_f1_mean": float(results.loc[best, "mean_test_score"]),
        "cv_macro_f1_std": float(results.loc[best, "std_test_score"]),
        "train_macro_f1_mean": float(results.loc[best, "mean_train_score"]),
        "fold_scores": folds,
        "seconds": round(elapsed, 1),
    }
    return record, searcher.best_estimator_


def report(record: dict) -> None:
    gap = record["train_macro_f1_mean"] - record["cv_macro_f1_mean"]
    print(f"\n[{record['label']}] {record['search']} ({record['n_candidates']} candidate(s), {record['seconds']}s)")
    print(f"  CV Macro F1  : {record['cv_macro_f1_mean']:.4f} ± {record['cv_macro_f1_std']:.4f}")
    print(f"  Train Macro F1: {record['train_macro_f1_mean']:.4f}   (train - CV gap {gap:+.4f})")
    print(f"  Folds        : {[round(s, 4) for s in record['fold_scores']]}")
    if record["best_params"]:
        print(f"  Best params  : {record['best_params']}")


def save_candidate(key: str, estimator) -> None:
    joblib.dump(estimator, CANDIDATES_DIR / f"{key}.joblib")


def main() -> None:
    ensure_dirs()
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, _, y_train, _ = load_split()  # test split deliberately ignored
    feature_params = load_selected_feature_params()
    print(f"Training rows: {len(X_train)} | feature set: {feature_params}")

    records, initial_records = [], []

    # Stage 7 - baselines
    rec, est = cv_only("dummy", make_pipeline(dummy_model(), feature_params), X_train, y_train,
                       "none (fixed strategy)")
    records.append(rec); report(rec); save_candidate("dummy", est)

    rec, est = search("logistic_regression", make_pipeline(logistic_model(), feature_params, scale_numeric=True),
                      LOGISTIC_GRID, X_train, y_train, kind="grid")
    records.append(rec); report(rec); save_candidate("logistic_regression", est)

    # Stages 8-10 - initial (untuned) run, then tuning, for each ensemble
    ensembles = [
        ("random_forest", random_forest_model, RANDOM_FOREST_SPACE, 30),
        ("xgboost", xgboost_model, XGBOOST_SPACE, 30),
        ("catboost", catboost_model, CATBOOST_SPACE, 20),
    ]
    best_params = {}
    for key, factory, space, n_iter in ensembles:
        init_rec, _ = cv_only(key, make_pipeline(factory(), feature_params), X_train, y_train,
                              "initial model, library defaults")
        init_rec["label"] = f"{MODEL_LABELS[key]} (initial)"
        initial_records.append(init_rec); report(init_rec)

        rec, est = search(key, make_pipeline(factory(), feature_params), space, X_train, y_train,
                          kind="random", n_iter=n_iter)
        records.append(rec); report(rec); save_candidate(key, est)
        best_params[key] = strip_prefix(rec["best_params"])

    # Stage 11 - stacking on the tuned base learners
    stack = make_pipeline(
        stacking_model(best_params["random_forest"], best_params["xgboost"], best_params["catboost"]),
        feature_params,
    )
    rec, est = search("stacking", stack, STACKING_GRID, X_train, y_train, kind="grid")
    records.append(rec); report(rec); save_candidate("stacking", est)

    # Stage 12 - consolidated tuning and CV tables
    summary = pd.DataFrame(records)
    summary["train_minus_cv_gap"] = summary["train_macro_f1_mean"] - summary["cv_macro_f1_mean"]
    summary.drop(columns=["fold_scores"]).to_csv(TUNING_DIR / "tuning_summary.csv", index=False)

    folds = pd.DataFrame(
        {r["label"]: r["fold_scores"] for r in initial_records + records},
        index=[f"fold_{i + 1}" for i in range(make_cv().get_n_splits())],
    ).T
    folds["mean"] = folds.mean(axis=1)
    folds["std"] = folds.iloc[:, :-1].std(axis=1, ddof=0)
    folds.round(4).to_csv(TUNING_DIR / "cv_fold_scores.csv")

    initial_vs_tuned = pd.DataFrame(
        [
            {
                "model": MODEL_LABELS[i["model"]],
                "initial_cv_macro_f1": i["cv_macro_f1_mean"],
                "initial_cv_std": i["cv_macro_f1_std"],
                "tuned_cv_macro_f1": t["cv_macro_f1_mean"],
                "tuned_cv_std": t["cv_macro_f1_std"],
                "improvement": t["cv_macro_f1_mean"] - i["cv_macro_f1_mean"],
            }
            for i in initial_records
            for t in records
            if t["model"] == i["model"]
        ]
    )
    initial_vs_tuned.round(4).to_csv(TUNING_DIR / "initial_vs_tuned.csv", index=False)

    with open(TUNING_DIR / "tuning_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"initial": initial_records, "tuned": records}, handle, indent=2, default=_jsonable)

    print("\nFold-wise CV Macro F1:")
    print(folds.round(4).to_string())
    print("\nInitial vs tuned:")
    print(initial_vs_tuned.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
