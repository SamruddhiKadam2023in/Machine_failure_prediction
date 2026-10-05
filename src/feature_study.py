"""Stages 5-6 - Train/test split, feature-set comparison and selection.

Order matters for leakage: the stratified split happens FIRST, and every
feature decision below uses the training split only (5-fold stratified CV).
The test split is saved to disk and not read again until final evaluation.

    python -m src.feature_study
"""

import json

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_validate

from config import CLASS_NAMES, FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET_COLUMN
from src.data_loading import load_raw_data
from src.feature_engineering import ENGINEERED_FEATURES, FEATURE_SETS, add_engineered_features
from src.models import make_cv, make_pipeline, random_forest_model, xgboost_model
from src.plotting import CLASS_COLORS, PRIMARY, TEXT, plt, save
from src.preprocessing import decode_target, make_train_test_split, prepare_model_data
from src.utils import ensure_dirs

STAGE5_DIR = REPORTS_DIR / "stage5"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
SELECTED_SET_PATH = STAGE5_DIR / "selected_feature_set.json"


def save_split() -> tuple:
    clean, _ = prepare_model_data(load_raw_data())
    X_train, X_test, y_train, y_test, udi_train, udi_test = make_train_test_split(clean)
    for X, y, udi, path in ((X_train, y_train, udi_train, TRAIN_PATH), (X_test, y_test, udi_test, TEST_PATH)):
        out = X.copy()
        out.insert(0, "UDI", udi)
        out[TARGET_COLUMN] = decode_target(y)
        out.to_csv(path, index=False)
    return X_train, X_test, y_train, y_test


def load_split():
    """Used by every later stage, so all of them see exactly the same rows."""
    from src.preprocessing import encode_target

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    feature_cols = [c for c in train.columns if c not in ("UDI", TARGET_COLUMN)]
    return (
        train[feature_cols],
        test[feature_cols],
        encode_target(train[TARGET_COLUMN]),
        encode_target(test[TARGET_COLUMN]),
    )


def split_summary(y_train, y_test) -> pd.DataFrame:
    rows = []
    for code, name in enumerate(CLASS_NAMES):
        n_tr, n_te = int((y_train == code).sum()), int((y_test == code).sum())
        rows.append(
            {
                "class": name,
                "train_count": n_tr,
                "train_%": round(100 * n_tr / len(y_train), 2),
                "test_count": n_te,
                "test_%": round(100 * n_te / len(y_test), 2),
            }
        )
    rows.append({"class": "Total", "train_count": len(y_train), "train_%": 100.0,
                 "test_count": len(y_test), "test_%": 100.0})
    return pd.DataFrame(rows)


def engineered_feature_views(X_train: pd.DataFrame, y_train) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlations and per-class medians of the engineered features (train only)."""
    enriched = add_engineered_features(X_train)
    numeric = enriched.drop(columns=["Type"])
    corr = numeric.corr().round(3)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True,
                linewidths=1, linecolor="white", cbar_kws={"label": "Pearson correlation", "shrink": 0.8}, ax=ax)
    ax.grid(False)
    ax.set_title("Correlation including engineered features (training split)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save(fig, FIGURES_DIR / "05_correlation_with_engineered.png")

    labels = pd.Series(decode_target(y_train), index=X_train.index, name=TARGET_COLUMN)
    medians = enriched[list(ENGINEERED_FEATURES)].groupby(labels).median().reindex(CLASS_NAMES).round(2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    frame = enriched.assign(**{TARGET_COLUMN: labels})
    for ax, feature in zip(axes, ENGINEERED_FEATURES):
        sns.boxplot(data=frame, x=TARGET_COLUMN, y=feature, hue=TARGET_COLUMN, order=CLASS_NAMES,
                    hue_order=CLASS_NAMES, palette=[CLASS_COLORS[c] for c in CLASS_NAMES], legend=False,
                    width=0.6, fliersize=2, linewidth=0.9, ax=ax)
        ax.set_title(feature)
        ax.set_xlabel("Failure type")
        ax.set_ylabel(feature)
    fig.suptitle("Engineered features by failure type (training split)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "05_engineered_features_by_class.png")
    return corr, medians


def mutual_information(X_train: pd.DataFrame, y_train) -> pd.DataFrame:
    """Filter-style relevance score of each candidate feature (train only)."""
    enriched = add_engineered_features(X_train)
    encoded = enriched.assign(Type=enriched["Type"].map({"L": 0, "M": 1, "H": 2}))
    discrete = [c == "Type" for c in encoded.columns]
    scores = mutual_info_classif(encoded, y_train, discrete_features=discrete, random_state=RANDOM_STATE)
    table = pd.DataFrame({"feature": encoded.columns, "mutual_information": scores})
    return table.sort_values("mutual_information", ascending=False).round(4).reset_index(drop=True)


def compare_feature_sets(X_train: pd.DataFrame, y_train) -> pd.DataFrame:
    """5-fold stratified CV Macro F1 for each feature set with two model families.

    Default hyperparameters (plus class balancing) are used on purpose: the
    comparison should reflect the features, not tuning luck. Tuning happens
    later on the selected set.
    """
    candidates = {
        "Random Forest": lambda: random_forest_model().set_params(class_weight="balanced"),
        "XGBoost": lambda: xgboost_model().set_params(class_weight="balanced"),
    }
    rows = []
    for set_name, params in FEATURE_SETS.items():
        for model_name, factory in candidates.items():
            result = cross_validate(make_pipeline(factory(), params), X_train, y_train, cv=make_cv(),
                                    scoring="f1_macro", n_jobs=2)
            scores = result["test_score"]
            n_features = len(make_pipeline(factory(), params).named_steps["features"]
                             .fit(X_train.head()).get_feature_names_out())
            rows.append({"feature_set": set_name, "model": model_name, "n_input_features": n_features,
                         "cv_macro_f1_mean": scores.mean(), "cv_macro_f1_std": scores.std(),
                         **{f"fold_{i + 1}": s for i, s in enumerate(scores)}})
            print(f"  {set_name:38s} {model_name:14s} {scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(rows)


def plot_feature_set_comparison(results: pd.DataFrame) -> None:
    sets = list(FEATURE_SETS)
    models = results["model"].unique()
    colors = {"Random Forest": PRIMARY, "XGBoost": "#eb6834"}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(sets))
    width = 0.36
    for i, model in enumerate(models):
        sub = results[results["model"] == model].set_index("feature_set").loc[sets]
        bars = ax.bar(x + (i - 0.5) * (width + 0.02), sub["cv_macro_f1_mean"], width, yerr=sub["cv_macro_f1_std"],
                      capsize=3, color=colors[model], label=model, error_kw={"elinewidth": 1, "ecolor": TEXT})
        for bar, mean in zip(bars, sub["cv_macro_f1_mean"]):
            ax.annotate(f"{mean:.3f}", (bar.get_x() + bar.get_width() / 2, 0.02), ha="center", va="bottom",
                        fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x, sets)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Macro F1 (5-fold CV, mean ± std)")
    ax.set_title("Feature-set comparison on the training split")
    ax.legend(title="Model", loc="lower right")
    save(fig, FIGURES_DIR / "05_feature_set_comparison.png")


def choose_feature_set(results: pd.DataFrame) -> dict:
    """Pre-defined rule: highest CV Macro F1 averaged over both model families."""
    summary = results.groupby("feature_set", sort=False)[["cv_macro_f1_mean", "n_input_features"]].mean()
    best = summary["cv_macro_f1_mean"].idxmax()
    return {"selected_feature_set": best, "params": {k: list(v) if isinstance(v, tuple) else v
                                                     for k, v in FEATURE_SETS[best].items()},
            "rule": "highest 5-fold CV Macro F1 averaged over Random Forest and XGBoost",
            "mean_cv_macro_f1_by_set": summary["cv_macro_f1_mean"].round(4).to_dict()}


def load_selected_feature_params() -> dict:
    with open(SELECTED_SET_PATH, encoding="utf-8") as handle:
        params = json.load(handle)["params"]
    params["drop_columns"] = tuple(params["drop_columns"])
    return params


def main() -> None:
    ensure_dirs()
    STAGE5_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = save_split()
    summary = split_summary(y_train, y_test)
    summary.to_csv(STAGE5_DIR / "split_summary.csv", index=False)
    print("Stage 6 - stratified split (test_size=0.20, random_state=42):")
    print(summary.to_string(index=False))

    corr, medians = engineered_feature_views(X_train, y_train)
    corr.to_csv(STAGE5_DIR / "correlation_with_engineered.csv")
    medians.to_csv(STAGE5_DIR / "engineered_feature_medians_by_class.csv")
    print("\nEngineered feature medians by class (train):")
    print(medians.to_string())
    print("\nHighest absolute correlations involving engineered features (train):")
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
    pairs = pairs[[any(f in pair for f in ENGINEERED_FEATURES) for pair in pairs.index]]
    print(pairs.abs().sort_values(ascending=False).head(6).round(3).to_string())

    mi = mutual_information(X_train, y_train)
    mi.to_csv(STAGE5_DIR / "mutual_information.csv", index=False)
    print("\nMutual information with the target (train):")
    print(mi.to_string(index=False))

    print("\nFeature-set comparison (5-fold stratified CV, Macro F1):")
    results = compare_feature_sets(X_train, y_train)
    results.round(4).to_csv(STAGE5_DIR / "feature_set_comparison.csv", index=False)
    plot_feature_set_comparison(results)

    choice = choose_feature_set(results)
    with open(SELECTED_SET_PATH, "w", encoding="utf-8") as handle:
        json.dump(choice, handle, indent=2)
    print("\nSelected:", json.dumps(choice, indent=2))


if __name__ == "__main__":
    main()
