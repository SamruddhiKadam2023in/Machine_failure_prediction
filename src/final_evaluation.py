"""Stages 13-17 - Test evaluation, misclassification analysis, comparison,
interpretation, final-model selection and saving.

This is the ONLY module that reads the test split. The selection rule is
fixed in advance and uses cross-validation scores from the training split,
so the test set measures performance but does not choose the model:

    Final model = highest mean 5-fold CV Macro F1 (Stage 12).

    python -m src.final_evaluation
"""

import json
import platform
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from config import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    FIGURES_DIR,
    FINAL_MODEL_PATH,
    METADATA_PATH,
    MODEL_INPUT_COLUMNS,
    NUMERICAL_FEATURES,
    PREPROCESSING_PATH,
    PRODUCT_TYPES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
)
from src.evaluation import (
    compute_metrics,
    confusion_pairs,
    per_class_table,
    plot_confusion_matrices,
    plot_roc_curves,
    text_report,
)
from src.feature_engineering import add_engineered_features
from src.feature_study import load_selected_feature_params, load_split
from src.plotting import PRIMARY, TEXT, plt, save
from src.preprocessing import decode_target
from src.tuning import CANDIDATES_DIR, MODEL_LABELS, TUNING_DIR
from src.utils import ensure_dirs, file_sha256

EVAL_DIR = REPORTS_DIR / "evaluation"
MODEL_ORDER = ["dummy", "logistic_regression", "random_forest", "xgboost", "catboost", "stacking"]
TREE_MODELS = {"random_forest", "xgboost", "catboost"}


# --------------------------------------------------------------- stage 13 ---

def evaluate_all(X_test, y_test, tuning: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows, fitted, per_class_rows = [], {}, []
    for key in MODEL_ORDER:
        pipeline = joblib.load(CANDIDATES_DIR / f"{key}.joblib")
        fitted[key] = pipeline
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        cv = tuning.loc[key]
        rows.append({"key": key, "model": MODEL_LABELS[key],
                     "cv_macro_f1": cv["cv_macro_f1_mean"], "cv_macro_f1_std": cv["cv_macro_f1_std"],
                     **{f"test_{k}": v for k, v in metrics.items()}})

        with open(EVAL_DIR / f"classification_report_{key}.txt", "w", encoding="utf-8") as handle:
            handle.write(f"{MODEL_LABELS[key]} - test set (n={len(y_test)})\n\n{text_report(y_test, y_pred)}")
        cm = plot_confusion_matrices(y_test, y_pred, MODEL_LABELS[key], f"13_confusion_matrix_{key}")
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(EVAL_DIR / f"confusion_matrix_{key}.csv")
        table = per_class_table(y_test, y_pred)
        table.insert(0, "model", MODEL_LABELS[key])
        per_class_rows.append(table)

        print(f"\n{MODEL_LABELS[key]}")
        print(f"  CV Macro F1 {cv['cv_macro_f1_mean']:.4f} | test: acc {metrics['accuracy']:.4f}  "
              f"macroP {metrics['macro_precision']:.4f}  macroR {metrics['macro_recall']:.4f}  "
              f"macroF1 {metrics['macro_f1']:.4f}  wF1 {metrics['weighted_f1']:.4f}  "
              f"AUC(macro) {metrics['macro_roc_auc']:.4f}  AUC(weighted) {metrics['weighted_roc_auc']:.4f}")
    return pd.DataFrame(rows).set_index("key"), fitted, pd.concat(per_class_rows, ignore_index=True)


def plot_per_class_f1(per_class: pd.DataFrame) -> None:
    pivot = per_class.pivot(index="model", columns="class", values="f1")[CLASS_NAMES]
    pivot = pivot.reindex([MODEL_LABELS[k] for k in MODEL_ORDER])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, linewidths=1, linecolor="white",
                cbar_kws={"label": "F1-score", "shrink": 0.85}, ax=ax)
    ax.grid(False)
    ax.set_title("Per-class F1-score on the test set")
    ax.set_xlabel("Failure type")
    ax.set_ylabel("Model")
    save(fig, FIGURES_DIR / "13_per_class_f1_all_models.png")


# --------------------------------------------------------------- stage 15 ---

def select_final(comparison: pd.DataFrame) -> str:
    """Pre-defined rule: highest mean CV Macro F1 on the training split."""
    return comparison["cv_macro_f1"].idxmax()


def comparison_table(comparison: pd.DataFrame, final_key: str) -> pd.DataFrame:
    cols = {
        "cv_macro_f1": "CV Macro F1 (mean)",
        "cv_macro_f1_std": "CV Macro F1 (std)",
        "test_accuracy": "Test Accuracy",
        "test_macro_precision": "Test Precision (macro)",
        "test_macro_recall": "Test Recall (macro)",
        "test_macro_f1": "Test F1 (macro)",
        "test_weighted_f1": "Test F1 (weighted)",
        "test_macro_roc_auc": "Test ROC-AUC (macro OvR)",
        "test_weighted_roc_auc": "Test ROC-AUC (weighted OvR)",
    }
    table = comparison.set_index("model")[list(cols)].rename(columns=cols)
    best = table.loc[[MODEL_LABELS[final_key]]].copy()
    best.index = [f"Best Model ({MODEL_LABELS[final_key]})"]
    return pd.concat([table, best]).round(4)


def plot_model_comparison(comparison: pd.DataFrame) -> None:
    models = comparison.loc[MODEL_ORDER]
    labels = [MODEL_LABELS[k].replace("Baseline (", "").replace(")", "").replace(", most frequent", "")
              for k in MODEL_ORDER]
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars_cv = ax.bar(x - width / 2 - 0.01, models["cv_macro_f1"], width, yerr=models["cv_macro_f1_std"],
                     capsize=3, color=PRIMARY, label="CV Macro F1 (train, 5-fold)",
                     error_kw={"elinewidth": 1, "ecolor": TEXT})
    bars_test = ax.bar(x + width / 2 + 0.01, models["test_macro_f1"], width, color="#eb6834",
                       label="Test Macro F1 (held-out)")
    for bars, errors in ((bars_cv, models["cv_macro_f1_std"].to_numpy()), (bars_test, np.zeros(len(models)))):
        for bar, err in zip(bars, errors):
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + err),
                        ha="center", va="bottom", fontsize=8.5, color=TEXT, xytext=(0, 3),
                        textcoords="offset points")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1")
    ax.set_title("Model comparison: cross-validation vs held-out test Macro F1")
    ax.legend(loc="upper left")
    save(fig, FIGURES_DIR / "15_model_comparison_macro_f1.png")


# --------------------------------------------------------------- stage 14 ---

def misclassification_analysis(pipeline, X_test, y_test, final_label: str) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    enriched = add_engineered_features(X_test).reset_index(drop=True)
    enriched.insert(0, "actual", decode_target(y_test))
    enriched.insert(1, "predicted", decode_target(y_pred))
    enriched.insert(2, "confidence_in_prediction", y_proba.max(axis=1).round(4))
    enriched.insert(3, "probability_of_actual_class", y_proba[np.arange(len(y_test)), y_test].round(4))
    wrong = enriched[enriched["actual"] != enriched["predicted"]].copy()
    wrong.to_csv(REPORTS_DIR / "misclassification_analysis.csv", index=False)

    per_class = per_class_table(y_test, y_pred)
    per_class["tier"] = pd.cut(per_class["f1"], bins=[-0.01, 0.6, 0.85, 1.0], labels=["Low", "Medium", "High"])
    per_class.round(4).to_csv(EVAL_DIR / "final_model_per_class.csv", index=False)

    cm = plot_confusion_matrices(y_test, y_pred, final_label, "14_confusion_matrix_final_model")
    pairs = confusion_pairs(cm)
    pairs.to_csv(EVAL_DIR / "final_model_confusion_pairs.csv", index=False)

    # Compare misclassified rows with correctly classified rows of the same
    # actual class and of the predicted class, on every feature.
    correct = enriched[enriched["actual"] == enriched["predicted"]]
    feature_cols = NUMERICAL_FEATURES + ["Temp difference [K]", "Power [W]", "Strain [min*Nm]"]
    profile_rows = []
    for _, pair in pairs.iterrows():
        subset = wrong[(wrong["actual"] == pair["actual"]) & (wrong["predicted"] == pair["predicted"])]
        for label, frame in (
            (f"misclassified {pair['actual']} -> {pair['predicted']}", subset),
            (f"correct {pair['actual']}", correct[correct["actual"] == pair["actual"]]),
            (f"correct {pair['predicted']}", correct[correct["actual"] == pair["predicted"]]),
        ):
            row = {"pair": f"{pair['actual']} -> {pair['predicted']}", "group": label, "n": len(frame)}
            row.update(frame[feature_cols].median().round(2).to_dict())
            profile_rows.append(row)
    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(EVAL_DIR / "misclassification_feature_profiles.csv", index=False)

    print("\nFinal model per-class results:")
    print(per_class.round(4).to_string(index=False))
    print("\nConfusion pairs (Actual -> Predicted):")
    print(pairs.to_string(index=False))
    print(f"\nMisclassified test rows: {len(wrong)} of {len(enriched)}")
    print("\nMisclassified rows (with engineered features):")
    show = ["actual", "predicted", "confidence_in_prediction", "probability_of_actual_class", "Type"] + feature_cols
    print(wrong[show].round(2).to_string(index=False))
    print("\nMedian feature profiles per confusion pair:")
    print(profiles.to_string(index=False))
    return {"per_class": per_class, "pairs": pairs, "n_wrong": len(wrong)}


# --------------------------------------------------------------- stage 16 ---

def class_boundary_evidence(X_train, y_train) -> pd.DataFrame:
    """Where each failure class starts in the TRAINING data, on the feature
    that separates it. Used to explain test-set confusions with evidence."""
    d = add_engineered_features(X_train)
    d["cls"] = decode_target(y_train)
    nf = d[d["cls"] == "No Failure"]
    rows = []
    band = d[d["Tool wear [min]"].between(198, 246)]
    rows.append({"class": "TWF", "evidence": "share of training rows with tool wear 198-246 min that are TWF",
                 "value": round(100 * (band["cls"] == "TWF").mean(), 1), "unit": "%",
                 "detail": str(band["cls"].value_counts().to_dict())})
    hdf = d[d["cls"] == "HDF"]
    rows.append({"class": "HDF", "evidence": "max temp difference / max rpm of HDF rows",
                 "value": f"{hdf['Temp difference [K]'].max():.1f} K / {hdf['Rotational speed [rpm]'].max():.0f} rpm",
                 "unit": "", "detail": "No Failure rows with temp diff <= 8.6 K and rpm < 1380: "
                 f"{int(((nf['Temp difference [K]'] <= 8.6) & (nf['Rotational speed [rpm]'] < 1380)).sum())}"})
    power = d.loc[d["cls"] == "PWF", "Power [W]"]
    rows.append({"class": "PWF", "evidence": "PWF power (low-group max / high-group min) vs No Failure power range",
                 "value": f"{power[power < 6000].max():.0f} W / {power[power > 6000].min():.0f} W",
                 "unit": "", "detail": f"No Failure: {nf['Power [W]'].min():.0f}-{nf['Power [W]'].max():.0f} W"})
    for product in PRODUCT_TYPES:
        osf = d[(d["cls"] == "OSF") & (d["Type"] == product)]["Strain [min*Nm]"]
        rows.append({"class": "OSF", "evidence": f"Type {product}: min OSF strain vs max No Failure strain",
                     "value": f"{osf.min():.0f}" if len(osf) else "no OSF rows", "unit": "min*Nm",
                     "detail": f"n OSF={len(osf)}; No Failure max "
                               f"{nf.loc[nf['Type'] == product, 'Strain [min*Nm]'].max():.0f}"})
    table = pd.DataFrame(rows)
    table.to_csv(EVAL_DIR / "class_boundary_evidence.csv", index=False)
    print("\nClass boundaries in the training data (explains the confusion pairs):")
    print(table.to_string(index=False))
    return table


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def model_importance(pipeline: Pipeline, key: str) -> pd.DataFrame | None:
    """Native impurity/gain importance for single tree ensembles."""
    if key not in TREE_MODELS:
        return None
    model = pipeline.named_steps["model"]
    values = model.get_feature_importance() if key == "catboost" else model.feature_importances_
    values = np.asarray(values, dtype=float)
    values = values / values.sum()
    return pd.DataFrame({"feature": transformed_feature_names(pipeline), "native_importance": values})


def permutation_table(pipeline: Pipeline, X_test, y_test) -> pd.DataFrame:
    """Model-agnostic: drop in test Macro F1 when one input column is shuffled.

    Computed on the raw input columns (Type + 5 sensors), because those are
    what an operator controls; engineered features are recomputed from the
    shuffled column inside the pipeline.
    """
    result = permutation_importance(pipeline, X_test, y_test, scoring="f1_macro", n_repeats=10,
                                    random_state=RANDOM_STATE, n_jobs=1)
    return pd.DataFrame({"input_feature": X_test.columns, "macro_f1_drop_mean": result.importances_mean,
                         "macro_f1_drop_std": result.importances_std}).sort_values(
        "macro_f1_drop_mean", ascending=False).reset_index(drop=True)


def shap_explainer_source(pipeline: Pipeline, key: str) -> tuple[object, str]:
    """Tree model to explain with SHAP. For stacking, explain its strongest
    tree base learner (the meta-learner is a linear model on probabilities,
    so SHAP on the base learner is the interpretable part)."""
    model = pipeline.named_steps["model"]
    if key in TREE_MODELS:
        return model, MODEL_LABELS[key]
    if key == "stacking":
        names = [name for name, _ in model.estimators]
        coef_mass = np.abs(model.final_estimator_.coef_).reshape(len(CLASS_NAMES), len(names), -1).sum(axis=(0, 2))
        chosen = names[int(np.argmax(coef_mass))]
        return dict(zip(names, model.estimators_))[chosen], f"{chosen} base learner inside Stacking"
    raise ValueError(f"No SHAP explainer defined for {key}")


def shap_analysis(pipeline: Pipeline, key: str, X_test) -> pd.DataFrame:
    tree_model, source = shap_explainer_source(pipeline, key)
    transformed = pipeline.named_steps["preprocess"].transform(pipeline.named_steps["features"].transform(X_test))
    names = transformed_feature_names(pipeline)
    explainer = shap.TreeExplainer(tree_model)
    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3 and values.shape[0] == len(CLASS_NAMES) and values.shape[-1] != len(CLASS_NAMES):
        values = np.transpose(values, (1, 2, 0))  # -> (samples, features, classes)
    mean_abs = pd.DataFrame(np.abs(values).mean(axis=0), index=names, columns=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    order = mean_abs.sum(axis=1).sort_values(ascending=False).index
    sns.heatmap(mean_abs.loc[order], annot=True, fmt=".2f", cmap="Blues", linewidths=1, linecolor="white",
                cbar_kws={"label": "Mean |SHAP value| (log-odds)", "shrink": 0.85}, ax=ax)
    ax.grid(False)
    ax.set_title(f"SHAP: which features drive each failure type\n({source}, test set)")
    ax.set_xlabel("Failure type (model output)")
    ax.set_ylabel("Feature")
    save(fig, FIGURES_DIR / "16_shap_mean_abs_by_class.png")

    for cls in [c for c in CLASS_NAMES if c != "No Failure"]:
        idx = CLASS_NAMES.index(cls)
        plt.figure()
        shap.summary_plot(values[:, :, idx], transformed, feature_names=names, show=False, plot_size=(8, 5),
                          max_display=len(names))
        fig = plt.gcf()
        fig.axes[0].set_title(f"SHAP values for class {cls} ({CLASS_DESCRIPTIONS[cls]})", loc="left")
        fig.axes[0].set_xlabel(f"SHAP value (impact on {cls} log-odds)")
        save(fig, FIGURES_DIR / f"16_shap_beeswarm_{cls}.png")

    mean_abs.round(4).to_csv(EVAL_DIR / "shap_mean_abs_by_class.csv")
    print(f"\nSHAP source: {source}")
    print(mean_abs.loc[order].round(3).to_string())
    return mean_abs


def plot_importance(importance: pd.DataFrame, final_label: str) -> None:
    has_native = "native_importance" in importance.columns and importance["native_importance"].notna().any()
    ncols = 2 if has_native else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    axes = np.atleast_1d(axes)
    perm = importance.dropna(subset=["macro_f1_drop_mean"]).sort_values("macro_f1_drop_mean")
    axes[0].barh(perm["input_feature"], perm["macro_f1_drop_mean"], xerr=perm["macro_f1_drop_std"],
                 color=PRIMARY, height=0.6, error_kw={"elinewidth": 1, "ecolor": TEXT})
    axes[0].set_title("Permutation importance (raw inputs)")
    axes[0].set_xlabel("Drop in test Macro F1 when shuffled")
    axes[0].set_ylabel("Input feature")
    if has_native:
        native = importance.dropna(subset=["native_importance"]).sort_values("native_importance")
        axes[1].barh(native["feature"], native["native_importance"], color="#eb6834", height=0.6)
        axes[1].set_title("Model-native importance (normalised)")
        axes[1].set_xlabel("Share of total importance")
        axes[1].set_ylabel("Model feature (after preprocessing)")
    fig.suptitle(f"Feature importance — {final_label}", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "feature_importance.png")


# --------------------------------------------------------------- stage 17 ---

def save_final(pipeline: Pipeline, key: str, X_train, X_test, y_test, comparison: pd.DataFrame,
               tuning_params: dict) -> None:
    joblib.dump(pipeline, FINAL_MODEL_PATH)
    preprocessing = Pipeline([("features", pipeline.named_steps["features"]),
                              ("preprocess", pipeline.named_steps["preprocess"])])
    joblib.dump(preprocessing, PREPROCESSING_PATH)

    reloaded = joblib.load(FINAL_MODEL_PATH)
    same_pred = np.array_equal(reloaded.predict(X_test), pipeline.predict(X_test))
    same_proba = np.allclose(reloaded.predict_proba(X_test), pipeline.predict_proba(X_test))
    reloaded_pre = joblib.load(PREPROCESSING_PATH)
    same_transform = np.allclose(reloaded_pre.transform(X_test),
                                 pipeline.named_steps["preprocess"].transform(
                                     pipeline.named_steps["features"].transform(X_test)))
    if not (same_pred and same_proba and same_transform):
        raise RuntimeError("Reloaded model does not reproduce the in-memory model's output.")

    row = comparison.loc[key]
    metadata = {
        "model_key": key,
        "model_name": MODEL_LABELS[key],
        "selection_rule": "highest mean 5-fold stratified CV Macro F1 on the training split",
        "class_names": CLASS_NAMES,
        "class_descriptions": CLASS_DESCRIPTIONS,
        "input_columns": MODEL_INPUT_COLUMNS,
        "product_types": PRODUCT_TYPES,
        "feature_engineering": load_selected_feature_params() | {"drop_columns": list(
            load_selected_feature_params()["drop_columns"])},
        "best_params": tuning_params,
        "training_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUMERICAL_FEATURES},
        "training_medians": {c: float(X_train[c].median()) for c in NUMERICAL_FEATURES},
        "training_rows": int(len(X_train)),
        "cv_macro_f1_mean": float(row["cv_macro_f1"]),
        "cv_macro_f1_std": float(row["cv_macro_f1_std"]),
        "test_metrics": {k.replace("test_", ""): float(v) for k, v in row.items() if k.startswith("test_")},
        "random_state": RANDOM_STATE,
        "dataset_sha256": file_sha256(RAW_DATA_PATH),
        "python": platform.python_version(),
        "library_versions": {p: version(p) for p in ["scikit-learn", "xgboost", "catboost", "pandas", "numpy",
                                                    "joblib", "shap", "streamlit"]},
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"\nSaved {FINAL_MODEL_PATH.name}, {PREPROCESSING_PATH.name}, {METADATA_PATH.name}")
    print(f"Reload check: predictions identical={same_pred}, probabilities identical={same_proba}, "
          f"preprocessing identical={same_transform}")


def main() -> None:
    ensure_dirs()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()
    tuning = pd.read_csv(TUNING_DIR / "tuning_summary.csv").set_index("model")
    with open(TUNING_DIR / "tuning_summary.json", encoding="utf-8") as handle:
        tuned_params = {r["model"]: r["best_params"] for r in json.load(handle)["tuned"]}

    print(f"Stage 13 - evaluating every tuned model on the untouched test set (n={len(y_test)})")
    comparison, fitted, per_class = evaluate_all(X_test, y_test, tuning)
    per_class.round(4).to_csv(EVAL_DIR / "per_class_metrics_all_models.csv", index=False)
    plot_per_class_f1(per_class)

    final_key = select_final(comparison)
    final_label = MODEL_LABELS[final_key]
    print(f"\nStage 15 - selected by CV Macro F1: {final_label}")
    table = comparison_table(comparison, final_key)
    table.to_csv(REPORTS_DIR / "model_comparison.csv")
    print(table.to_string())
    plot_model_comparison(comparison)
    plot_roc_curves(y_test, fitted[final_key].predict_proba(X_test), final_label, "13_roc_curves_final_model")

    with open(REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(f"Final model: {final_label}\nSelected by: highest mean 5-fold CV Macro F1 (training split)\n\n")
        handle.write(text_report(y_test, fitted[final_key].predict(X_test)))

    print("\nStage 14 - misclassification analysis")
    misclassification_analysis(fitted[final_key], X_test, y_test, final_label)
    class_boundary_evidence(X_train, y_train)

    print("\nStage 16 - feature importance and interpretation")
    perm = permutation_table(fitted[final_key], X_test, y_test)
    native = model_importance(fitted[final_key], final_key)
    importance = perm.copy()
    if native is not None:
        importance = pd.concat([perm, native], axis=1)
    importance.round(5).to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    print(perm.round(4).to_string(index=False))
    if native is not None:
        print(native.sort_values("native_importance", ascending=False).round(4).to_string(index=False))
    plot_importance(importance, final_label)
    shap_analysis(fitted[final_key], final_key, X_test)

    print("\nStage 17 - saving the final pipeline")
    save_final(fitted[final_key], final_key, X_train, X_test, y_test, comparison, tuned_params[final_key])


if __name__ == "__main__":
    main()
