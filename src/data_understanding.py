"""Stage 2 - Dataset collection & understanding.

Computes every statistic directly from data/raw/ai4i2020.csv and saves
tables to reports/stage2/ and figures to figures/. Run from the project root:

    python -m src.data_understanding
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FAILURE_MODE_COLUMNS,
    FIGURES_DIR,
    ID_COLUMNS,
    MACHINE_FAILURE_COLUMN,
    NUMERICAL_FEATURES,
    RAW_DATA_PATH,
    REPORTS_DIR,
)
from src.data_loading import load_raw_data  # noqa: E402
from src.utils import ensure_dirs, file_sha256  # noqa: E402

STAGE2_DIR = REPORTS_DIR / "stage2"
BAR_COLOR = "#3B6EA8"
TEXT_COLOR = "#333333"

FEATURE_DICTIONARY = {
    "UDI": ("Identifier", "Unique row identifier, 1..10000"),
    "Product ID": ("Identifier", "Product serial; first letter is the quality variant"),
    "Type": ("Categorical", "Product quality variant: L (low), M (medium), H (high)"),
    "Air temperature [K]": ("Numerical sensor", "Ambient air temperature in kelvin"),
    "Process temperature [K]": ("Numerical sensor", "Process temperature in kelvin"),
    "Rotational speed [rpm]": ("Numerical sensor", "Spindle rotational speed"),
    "Torque [Nm]": ("Numerical sensor", "Torque in newton-metres"),
    "Tool wear [min]": ("Numerical sensor", "Cumulative minutes of tool use"),
    "Machine failure": ("Label (binary)", "1 if the machine failed, else 0"),
    "TWF": ("Label (failure mode)", "Tool wear failure"),
    "HDF": ("Label (failure mode)", "Heat dissipation failure"),
    "PWF": ("Label (failure mode)", "Power failure"),
    "OSF": ("Label (failure mode)", "Overstrain failure"),
    "RNF": ("Label (failure mode)", "Random failure"),
}


def build_feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        role, description = FEATURE_DICTIONARY[column]
        rows.append(
            {
                "column": column,
                "role": role,
                "dtype": str(df[column].dtype),
                "n_unique": df[column].nunique(),
                "n_missing": int(df[column].isna().sum()),
                "description": description,
            }
        )
    return pd.DataFrame(rows)


def duplicate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    without_ids = df.drop(columns=ID_COLUMNS)
    features_only = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    return pd.DataFrame(
        {
            "check": [
                "Exact duplicate rows (all columns)",
                "Duplicate rows ignoring UDI and Product ID",
                "Duplicate feature vectors (Type + 5 sensors)",
                "Duplicate UDI values",
                "Duplicate Product ID values",
            ],
            "count": [
                int(df.duplicated().sum()),
                int(without_ids.duplicated().sum()),
                int(features_only.duplicated().sum()),
                int(df["UDI"].duplicated().sum()),
                int(df["Product ID"].duplicated().sum()),
            ],
        }
    )


def failure_label_analysis(df: pd.DataFrame) -> dict:
    """Break down how 'Machine failure' relates to the five mode flags."""
    n_modes = df[FAILURE_MODE_COLUMNS].sum(axis=1)
    machine_failure = df[MACHINE_FAILURE_COLUMN]

    mode_counts = pd.DataFrame(
        {
            "flag": [MACHINE_FAILURE_COLUMN] + FAILURE_MODE_COLUMNS,
            "count": [int(df[c].sum()) for c in [MACHINE_FAILURE_COLUMN] + FAILURE_MODE_COLUMNS],
        }
    )
    mode_counts["percent_of_rows"] = (100 * mode_counts["count"] / len(df)).round(2)

    crosstab = pd.crosstab(
        machine_failure.rename("Machine failure"),
        n_modes.rename("number of mode flags set"),
    )

    def combo_name(row) -> str:
        active = [mode for mode in FAILURE_MODE_COLUMNS if row[mode] == 1]
        return "+".join(active) if active else "none"

    combos = df[FAILURE_MODE_COLUMNS].apply(combo_name, axis=1)
    combo_table = (
        pd.DataFrame({"Machine failure": machine_failure, "mode combination": combos})
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["Machine failure", "count"], ascending=[True, False])
    )

    inconsistencies = pd.DataFrame(
        {
            "case": [
                "Machine failure = 1, no mode flag set",
                "Machine failure = 0, a mode flag set",
                "Rows with more than one mode flag",
            ],
            "count": [
                int(((machine_failure == 1) & (n_modes == 0)).sum()),
                int(((machine_failure == 0) & (n_modes > 0)).sum()),
                int((n_modes > 1).sum()),
            ],
        }
    )
    return {
        "mode_counts": mode_counts,
        "crosstab": crosstab,
        "combos": combo_table,
        "inconsistencies": inconsistencies,
    }


def style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, color=TEXT_COLOR, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.set_axisbelow(True)


def annotate_bars(ax, bars, total: int) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}\n({100 * height / total:.2f}%)",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT_COLOR,
            xytext=(0, 2),
            textcoords="offset points",
        )


def plot_machine_failure(df: pd.DataFrame) -> None:
    counts = df[MACHINE_FAILURE_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(["No failure (0)", "Failure (1)"], counts.values, color=BAR_COLOR, width=0.55)
    annotate_bars(ax, bars, len(df))
    ax.set_ylim(0, counts.max() * 1.15)
    style_axis(ax, "Binary 'Machine failure' indicator", "Machine failure", "Number of records")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_machine_failure_imbalance.png", dpi=150)
    plt.close(fig)


def plot_failure_modes(mode_counts: pd.DataFrame, n_rows: int) -> None:
    modes = mode_counts[mode_counts["flag"] != MACHINE_FAILURE_COLUMN]
    labels = {
        "TWF": "TWF\nTool wear",
        "HDF": "HDF\nHeat dissipation",
        "PWF": "PWF\nPower",
        "OSF": "OSF\nOverstrain",
        "RNF": "RNF\nRandom",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar([labels[f] for f in modes["flag"]], modes["count"], color=BAR_COLOR, width=0.6)
    annotate_bars(ax, bars, n_rows)
    ax.set_ylim(0, modes["count"].max() * 1.2)
    style_axis(
        ax,
        "Failure-mode flags set in the raw data (flags can overlap)",
        "Failure mode",
        "Number of records with flag = 1",
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_failure_mode_counts.png", dpi=150)
    plt.close(fig)


def plot_product_type(df: pd.DataFrame) -> None:
    counts = df["Type"].value_counts().reindex(["L", "M", "H"])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(["L (low)", "M (medium)", "H (high)"], counts.values, color=BAR_COLOR, width=0.55)
    annotate_bars(ax, bars, len(df))
    ax.set_ylim(0, counts.max() * 1.15)
    style_axis(ax, "Product quality variant (Type)", "Type", "Number of records")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_product_type_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    print(f"Source file: {RAW_DATA_PATH.relative_to(RAW_DATA_PATH.parents[2])}")
    print(f"SHA-256: {file_sha256(RAW_DATA_PATH)}")
    print(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns\n")

    feature_dict = build_feature_dictionary(df)
    feature_dict.to_csv(STAGE2_DIR / "feature_dictionary.csv", index=False)
    print("Feature dictionary:")
    print(feature_dict[["column", "role", "dtype", "n_unique", "n_missing"]].to_string(index=False))

    summary = df[NUMERICAL_FEATURES].describe().T
    summary["skew"] = df[NUMERICAL_FEATURES].skew()
    summary = summary.round(3)
    summary.to_csv(STAGE2_DIR / "numerical_summary.csv")
    print("\nNumerical summary:")
    print(summary.to_string())

    type_counts = df["Type"].value_counts().rename("count").to_frame()
    type_counts["percent"] = (100 * type_counts["count"] / len(df)).round(2)
    type_counts.to_csv(STAGE2_DIR / "type_distribution.csv")
    print("\nType distribution:")
    print(type_counts.to_string())

    missing = df.isna().sum().rename("missing_count").to_frame()
    missing.to_csv(STAGE2_DIR / "missing_values.csv")
    print(f"\nTotal missing values: {int(missing['missing_count'].sum())}")

    duplicates = duplicate_analysis(df)
    duplicates.to_csv(STAGE2_DIR / "duplicate_analysis.csv", index=False)
    print("\nDuplicate analysis:")
    print(duplicates.to_string(index=False))

    id_checks = {
        "UDI equals 1..N in order": bool((df["UDI"] == range(1, len(df) + 1)).all()),
        "Product ID prefix equals Type": bool((df["Product ID"].str[0] == df["Type"]).all()),
    }
    print("\nIdentifier checks:", id_checks)

    labels = failure_label_analysis(df)
    labels["mode_counts"].to_csv(STAGE2_DIR / "failure_flag_counts.csv", index=False)
    labels["crosstab"].to_csv(STAGE2_DIR / "machine_failure_vs_mode_count.csv")
    labels["combos"].to_csv(STAGE2_DIR / "failure_mode_combinations.csv", index=False)
    labels["inconsistencies"].to_csv(STAGE2_DIR / "label_inconsistencies.csv", index=False)

    print("\nFailure flag counts:")
    print(labels["mode_counts"].to_string(index=False))
    print("\nMachine failure vs number of mode flags set:")
    print(labels["crosstab"].to_string())
    print("\nFailure-mode combinations:")
    print(labels["combos"].to_string(index=False))
    print("\nLabel inconsistencies:")
    print(labels["inconsistencies"].to_string(index=False))

    failures = int(df[MACHINE_FAILURE_COLUMN].sum())
    print(
        f"\nImbalance ratio (no failure : failure) = "
        f"{(len(df) - failures) / failures:.1f} : 1"
    )

    plot_machine_failure(df)
    plot_failure_modes(labels["mode_counts"], len(df))
    plot_product_type(df)
    print("\nSaved tables to reports/stage2/ and figures to figures/.")


if __name__ == "__main__":
    main()
