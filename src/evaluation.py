"""Stage 13 - Evaluation helpers shared by every model.

All metrics are computed from real predictions; nothing is estimated.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from config import CLASS_NAMES, FIGURES_DIR
from src.plotting import CLASS_COLORS, MUTED, plt, save

LABELS = list(range(len(CLASS_NAMES)))


def ovr_roc_auc(y_true, y_proba) -> tuple[float, float, list[float]]:
    """One-vs-Rest ROC-AUC computed class by class.

    Each class's AUC ranks that class's score column against all other rows,
    which is the definition of OvR AUC. Computing it per column (instead of
    sklearn's multi_class='ovr' shortcut) also supports models whose class
    scores do not sum to 1, e.g. CatBoost with loss MultiClassOneVsAll; for
    models whose rows do sum to 1 the result is identical to sklearn's.
    Returns (macro, support-weighted, per-class list).
    """
    y_true = np.asarray(y_true)
    per_class = [roc_auc_score(y_true == code, y_proba[:, code]) for code in LABELS]
    support = np.array([(y_true == code).sum() for code in LABELS], dtype=float)
    return float(np.mean(per_class)), float(np.average(per_class, weights=support)), per_class


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Headline metrics. ROC-AUC is One-vs-Rest on predicted probabilities."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
        "macro_roc_auc": np.nan,
        "weighted_roc_auc": np.nan,
    }
    if y_proba is not None:
        metrics["macro_roc_auc"], metrics["weighted_roc_auc"], _ = ovr_roc_auc(y_true, y_proba)
    return metrics


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return pd.DataFrame(
        {"class": CLASS_NAMES, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )


def text_report(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, labels=LABELS, target_names=CLASS_NAMES, digits=4, zero_division=0
    )


def plot_confusion_matrices(y_true, y_pred, model_name: str, file_stem: str) -> np.ndarray:
    """Raw counts and row-normalised (recall) confusion matrices side by side."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, data, fmt, title, cbar_label in (
        (axes[0], cm, "d", "Counts", "Number of test records"),
        (axes[1], cm_norm, ".2f", "Normalised by true class (row = recall)", "Share of true class"),
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    linewidths=1, linecolor="white", cbar_kws={"label": cbar_label, "shrink": 0.85}, ax=ax,
                    vmin=0, vmax=1 if fmt == ".2f" else None)
        ax.grid(False)
        ax.set_title(title)
        ax.set_xlabel("Predicted failure type")
        ax.set_ylabel("Actual failure type")
    fig.suptitle(f"{model_name} — confusion matrix on the test set (n = {len(y_true)})",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, FIGURES_DIR / f"{file_stem}.png")
    return cm


def plot_roc_curves(y_true, y_proba, model_name: str, file_stem: str) -> None:
    """One-vs-Rest ROC curve per class."""
    y_bin = label_binarize(y_true, classes=LABELS)
    _, _, per_class_auc = ovr_roc_auc(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6.8, 6))
    for code, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, code], y_proba[:, code])
        auc = per_class_auc[code]
        ax.plot(fpr, tpr, linewidth=2, color=CLASS_COLORS[name] if name != "No Failure" else "#52514e",
                label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=MUTED, label="Chance")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_name} — One-vs-Rest ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, FIGURES_DIR / f"{file_stem}.png")


def confusion_pairs(cm: np.ndarray) -> pd.DataFrame:
    """Off-diagonal cells, most frequent first: 'Actual A -> Predicted B'."""
    rows = []
    for i, actual in enumerate(CLASS_NAMES):
        for j, predicted in enumerate(CLASS_NAMES):
            if i != j and cm[i, j] > 0:
                rows.append({"actual": actual, "predicted": predicted, "count": int(cm[i, j]),
                             "share_of_actual_class_%": round(100 * cm[i, j] / cm[i].sum(), 1)})
    table = pd.DataFrame(rows, columns=["actual", "predicted", "count", "share_of_actual_class_%"])
    return table.sort_values("count", ascending=False).reset_index(drop=True)
