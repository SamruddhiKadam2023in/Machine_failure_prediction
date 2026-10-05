"""Stage 4 - Exploratory data analysis.

Uses the cleaned 5-class data from Stage 3. EDA here is descriptive only:
no transformation is fitted, and modelling choices that could be tuned
(feature selection, hyperparameters) are made later with CV on the training
split, not from these plots.

    python -m src.eda
"""

import numpy as np
import pandas as pd
import seaborn as sns

from config import CLASS_NAMES, FIGURES_DIR, NUMERICAL_FEATURES, PRODUCT_TYPES, REPORTS_DIR, TARGET_COLUMN
from src.data_loading import load_raw_data
from src.plotting import CLASS_COLORS, MUTED, NEUTRAL, PRIMARY, TEXT, plt, save
from src.preprocessing import prepare_model_data
from src.utils import ensure_dirs

STAGE4_DIR = REPORTS_DIR / "stage4"
FAILURE_CLASSES = [c for c in CLASS_NAMES if c != "No Failure"]
SHORT = {
    "Air temperature [K]": "Air temp [K]",
    "Process temperature [K]": "Process temp [K]",
    "Rotational speed [rpm]": "Rot. speed [rpm]",
    "Torque [Nm]": "Torque [Nm]",
    "Tool wear [min]": "Tool wear [min]",
}


def plot_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET_COLUMN].value_counts().reindex(CLASS_NAMES)
    pct = 100 * counts / counts.sum()
    table = pd.DataFrame({"count": counts, "percent": pct.round(2)})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1]})
    for ax, classes, title in (
        (axes[0], CLASS_NAMES, "All classes"),
        (axes[1], FAILURE_CLASSES, "Failure classes only (zoomed)"),
    ):
        values = counts[classes]
        bars = ax.bar(classes, values, color=[CLASS_COLORS[c] for c in classes], width=0.6)
        for bar, cls in zip(bars, classes):
            ax.annotate(
                f"{counts[cls]}\n{pct[cls]:.2f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8.5, color=TEXT, xytext=(0, 2), textcoords="offset points",
            )
        ax.set_ylim(0, values.max() * 1.18)
        ax.set_title(title)
        ax.set_xlabel("Failure type")
        ax.set_ylabel("Number of records")
    fig.suptitle("Target class distribution after cleaning (n = %d)" % len(df), x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "04_class_distribution.png")
    return table


def plot_numerical_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, NUMERICAL_FEATURES):
        sns.histplot(df[col], bins=40, kde=True, color=PRIMARY, edgecolor="white", linewidth=0.4, ax=ax)
        ax.set_title(SHORT[col])
        ax.set_xlabel(col)
        ax.set_ylabel("Number of records")
    axes.flat[-1].axis("off")
    fig.suptitle("Distributions of the numerical sensor features", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "04_numerical_distributions.png")


def plot_boxplots_by_class(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    palette = [CLASS_COLORS[c] for c in CLASS_NAMES]
    for ax, col in zip(axes.flat, NUMERICAL_FEATURES):
        sns.boxplot(
            data=df, x=TARGET_COLUMN, y=col, hue=TARGET_COLUMN, order=CLASS_NAMES, hue_order=CLASS_NAMES,
            palette=palette, legend=False, width=0.6, fliersize=2, linewidth=0.9, ax=ax,
        )
        ax.set_title(f"{SHORT[col]} by failure type")
        ax.set_xlabel("Failure type")
        ax.set_ylabel(col)
    axes.flat[-1].axis("off")
    fig.suptitle("Sensor values by failure type (boxes: IQR, whiskers: 1.5×IQR)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "04_boxplots_by_failure_type.png")


def plot_correlation(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[NUMERICAL_FEATURES].corr(method="pearson")
    labels = [SHORT[c] for c in NUMERICAL_FEATURES]
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True,
        xticklabels=labels, yticklabels=labels, linewidths=1, linecolor="white",
        cbar_kws={"label": "Pearson correlation", "shrink": 0.8}, ax=ax,
    )
    ax.grid(False)
    ax.set_title("Correlation between numerical features")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    save(fig, FIGURES_DIR / "04_correlation_heatmap.png")
    return corr.round(3)


def plot_failure_rate_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Rate (%) of each failure class within each product quality variant."""
    rates = (
        pd.crosstab(df["Type"], df[TARGET_COLUMN], normalize="index")
        .reindex(index=PRODUCT_TYPES, columns=CLASS_NAMES)
        .fillna(0)
        * 100
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(PRODUCT_TYPES))
    width = 0.19
    for i, cls in enumerate(FAILURE_CLASSES):
        offset = (i - 1.5) * (width + 0.01)
        bars = ax.bar(x + offset, rates[cls], width=width, color=CLASS_COLORS[cls], label=cls)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=7.5, color=TEXT, xytext=(0, 1), textcoords="offset points")
    counts = df["Type"].value_counts()
    ax.set_xticks(x, [f"{t}  (n={counts[t]})" for t in PRODUCT_TYPES])
    ax.set_xlabel("Product quality variant (Type)")
    ax.set_ylabel("Share of records in that Type (%)")
    ax.set_title("Failure-mode rate within each product Type")
    ax.legend(title="Failure type", ncol=4, loc="upper right")
    ax.set_ylim(0, rates[FAILURE_CLASSES].to_numpy().max() * 1.3)
    save(fig, FIGURES_DIR / "04_failure_rate_by_type.png")
    return rates.round(3)


def plot_parameter_space(df: pd.DataFrame) -> None:
    """Small multiples: each failure mode highlighted against all other records.

    One panel per mode keeps colours unambiguous; overlaying four colours in one
    scatter is not colour-blind safe.
    """
    temp_diff = df["Process temperature [K]"] - df["Air temperature [K]"]
    views = [
        ("Rotational speed [rpm]", df["Rotational speed [rpm]"], "Torque [Nm]", df["Torque [Nm]"]),
        ("Tool wear [min]", df["Tool wear [min]"], "Torque [Nm]", df["Torque [Nm]"]),
        ("Rotational speed [rpm]", df["Rotational speed [rpm]"],
         "Process − air temperature [K]", temp_diff),
    ]
    fig, axes = plt.subplots(len(views), len(FAILURE_CLASSES), figsize=(16, 11.5), sharex="row", sharey="row")
    for r, (xlabel, xvals, ylabel, yvals) in enumerate(views):
        for c, cls in enumerate(FAILURE_CLASSES):
            ax = axes[r, c]
            is_cls = df[TARGET_COLUMN] == cls
            ax.scatter(xvals[~is_cls], yvals[~is_cls], s=3, color=NEUTRAL, alpha=0.25, linewidths=0,
                       label="All other records")
            ax.scatter(xvals[is_cls], yvals[is_cls], s=16, color=CLASS_COLORS[cls], edgecolors="white",
                       linewidths=0.4, label=f"{cls} (n={int(is_cls.sum())})")
            if r == 0:
                ax.set_title(cls)
            ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", fontsize=7.5, markerscale=1.2, handletextpad=0.3)
    fig.suptitle("Where each failure mode occurs in the operating space", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "04_parameter_space_by_failure_type.png")


def class_medians(df: pd.DataFrame) -> pd.DataFrame:
    table = df.groupby(TARGET_COLUMN)[NUMERICAL_FEATURES].median().reindex(CLASS_NAMES)
    table["Temp difference [K]"] = (
        df.assign(td=df["Process temperature [K]"] - df["Air temperature [K]"])
        .groupby(TARGET_COLUMN)["td"].median().reindex(CLASS_NAMES)
    )
    table["Power [W]"] = (
        df.assign(p=df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60)
        .groupby(TARGET_COLUMN)["p"].median().reindex(CLASS_NAMES)
    )
    table["Tool wear × torque [min·Nm]"] = (
        df.assign(s=df["Tool wear [min]"] * df["Torque [Nm]"])
        .groupby(TARGET_COLUMN)["s"].median().reindex(CLASS_NAMES)
    )
    return table.round(2)


def main() -> None:
    ensure_dirs()
    STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    df, _ = prepare_model_data(load_raw_data())

    dist = plot_class_distribution(df)
    dist.to_csv(STAGE4_DIR / "class_distribution.csv")
    print("Class distribution:\n", dist.to_string())

    plot_numerical_distributions(df)
    plot_boxplots_by_class(df)

    corr = plot_correlation(df)
    corr.to_csv(STAGE4_DIR / "correlation_matrix.csv")
    print("\nCorrelation matrix:\n", corr.to_string())

    rates = plot_failure_rate_by_type(df)
    rates.to_csv(STAGE4_DIR / "failure_rate_by_type_percent.csv")
    print("\nFailure-type rate (%) within each product Type:\n", rates.to_string())

    plot_parameter_space(df)

    medians = class_medians(df)
    medians.to_csv(STAGE4_DIR / "feature_medians_by_class.csv")
    print("\nMedian feature values by class:\n", medians.to_string())

    ranges = df.groupby(TARGET_COLUMN)[NUMERICAL_FEATURES].agg(["min", "max"]).reindex(CLASS_NAMES)
    ranges.to_csv(STAGE4_DIR / "feature_ranges_by_class.csv")
    print("\nFeature ranges by class:\n", ranges.round(1).to_string())
    print("\nSaved figures/04_*.png and reports/stage4/.")


if __name__ == "__main__":
    main()
