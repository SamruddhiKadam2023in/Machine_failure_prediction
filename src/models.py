"""Model definitions and hyperparameter search spaces (Stages 7-11).

Every model is wrapped in the same pipeline:

    FeatureEngineer -> ColumnTransformer (preprocessing) -> classifier

so feature engineering and preprocessing are refitted inside every CV fold
(no leakage) and the saved final pipeline carries them into deployment.
"""

import numpy as np
from catboost import CatBoostClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import CV_FOLDS, RANDOM_STATE
from src.feature_engineering import FeatureEngineer
from src.preprocessing import build_preprocessor


def make_cv() -> StratifiedKFold:
    """The single CV scheme shared by every model."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


class XGBClassifierCW(XGBClassifier):
    """XGBClassifier with a sklearn-style `class_weight` option.

    XGBoost's multi-class objective has no class_weight argument; the usual
    remedy is per-sample weights. Doing it inside fit() lets class weighting
    be tuned like any other hyperparameter and keeps it working inside CV and
    stacking, where passing sample weights from outside is awkward.
    """

    def __init__(self, *, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None, **kwargs):
        if self.class_weight is not None and sample_weight is None:
            sample_weight = compute_sample_weight(self.class_weight, y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)

    def get_xgb_params(self):
        # class_weight is handled in fit(); it is not a booster parameter.
        params = super().get_xgb_params()
        params.pop("class_weight", None)
        return params


class CatBoostClassifierFlat(CatBoostClassifier):
    """CatBoost returns multi-class predictions as an (n, 1) column; flatten
    them to (n,) like every other sklearn classifier so metrics, confusion
    matrices and the app all receive the same shape."""

    def predict(self, data, *args, **kwargs):
        return np.asarray(super().predict(data, *args, **kwargs)).ravel()


def make_pipeline(model, feature_params: dict, scale_numeric: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureEngineer(**feature_params)),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


# ------------------------------------------------------------ estimators ---

def dummy_model() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def logistic_model() -> LogisticRegression:
    return LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)


def random_forest_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)


def xgboost_model() -> XGBClassifierCW:
    return XGBClassifierCW(
        objective="multi:softprob",  # multi-class, returns class probabilities
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def catboost_model() -> CatBoostClassifierFlat:
    return CatBoostClassifierFlat(
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=0,
        thread_count=1,
        allow_writing_files=False,
    )


# ---------------------------------------------------------- search spaces ---
# Kept deliberately modest: the machine has 2 cores and the brief asks to
# avoid unnecessarily large searches.

LOGISTIC_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "model__class_weight": [None, "balanced"],
}

RANDOM_FOREST_SPACE = {
    "model__n_estimators": randint(200, 601),
    "model__max_depth": [None, 8, 12, 16, 24],
    "model__min_samples_split": randint(2, 11),
    "model__min_samples_leaf": randint(1, 6),
    "model__max_features": ["sqrt", "log2", 0.5, None],
    "model__class_weight": [None, "balanced", "balanced_subsample"],
}

XGBOOST_SPACE = {
    "model__n_estimators": randint(150, 701),
    "model__max_depth": randint(3, 9),
    "model__learning_rate": loguniform(0.02, 0.3),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__min_child_weight": [1, 2, 3, 5],
    "model__class_weight": [None, "balanced"],
}

CATBOOST_SPACE = {
    "model__iterations": randint(300, 901),
    "model__depth": randint(4, 9),
    "model__learning_rate": loguniform(0.02, 0.3),
    "model__l2_leaf_reg": loguniform(1, 10),
    "model__loss_function": ["MultiClass", "MultiClassOneVsAll"],
    # CatBoost expects the string "None" for no weighting; Python None fails.
    "model__auto_class_weights": ["None", "Balanced", "SqrtBalanced"],
}

STACKING_GRID = {
    "model__final_estimator__C": [0.1, 1.0, 10.0],
}


def strip_prefix(params: dict, prefix: str = "model__") -> dict:
    return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}


def stacking_model(rf_params: dict, xgb_params: dict, cat_params: dict) -> StackingClassifier:
    """Stack the three tuned ensembles under a logistic-regression meta-learner.

    Leakage control: the meta-learner is trained on *out-of-fold* class
    probabilities of the base learners (cv=StratifiedKFold inside
    StackingClassifier), never on predictions for rows a base learner was
    trained on. Probabilities are used because they carry each base learner's
    confidence, which matters for rare classes.
    """
    base_learners = [
        ("random_forest", random_forest_model().set_params(**rf_params)),
        ("xgboost", xgboost_model().set_params(**xgb_params)),
        ("catboost", catboost_model().set_params(**cat_params)),
    ]
    return StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
        cv=make_cv(),
        stack_method="predict_proba",
        n_jobs=1,
    )
