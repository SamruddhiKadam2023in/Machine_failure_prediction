"""Stage 3 - Data cleaning & preprocessing.

Two kinds of work live here, and the difference matters for leakage:

* Row-level cleaning (target construction, exclusions, validity checks).
  Each row is judged on its own values only; no statistic is computed across
  rows, so doing this before the train/test split cannot leak test information.

* The sklearn preprocessing transformer (build_preprocessor). It is only
  *defined* here. It is fitted inside each model pipeline on training data
  (or on the training folds during CV), never on the test set.

Run from the project root to regenerate the Stage 3 artefacts:

    python -m src.preprocessing
"""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    CLASS_NAMES,
    CLASS_TO_CODE,
    MACHINE_FAILURE_COLUMN,
    MODEL_INPUT_COLUMNS,
    NO_FAILURE_LABEL,
    NUMERICAL_FEATURES,
    PREDICTABLE_MODES,
    PROCESSED_DIR,
    PRODUCT_TYPES,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)

STAGE3_DIR = REPORTS_DIR / "stage3"
CLEAN_DATA_PATH = PROCESSED_DIR / "model_data.csv"

# Physical plausibility limits. Deliberately loose: they catch impossible values
# (sensor faults, unit mix-ups), not unusual-but-real operating points.
VALIDITY_RULES = {
    "Air temperature [K]": (250.0, 400.0),
    "Process temperature [K]": (250.0, 450.0),
    "Rotational speed [rpm]": (1.0, 10000.0),
    "Torque [Nm]": (0.0, 500.0),
    "Tool wear [min]": (0.0, 1000.0),
}


# ---------------------------------------------------------------- target ---

def build_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the 5-class 'Failure Type' column and exclude unmappable rows.

    Returns (clean_df, excluded_log). excluded_log has one row per excluded
    record with its UDI and the reason, so no row disappears silently.
    """
    n_predictable = df[PREDICTABLE_MODES].sum(axis=1)
    machine_failure = df[MACHINE_FAILURE_COLUMN]

    reasons = pd.Series("", index=df.index, dtype="object")
    # Order matters only for the log text; the three conditions do not overlap.
    reasons[(machine_failure == 0) & (df["RNF"] == 1)] = (
        "Contradictory labels: RNF=1 but Machine failure=0"
    )
    reasons[(machine_failure == 1) & (df[PREDICTABLE_MODES + ["RNF"]].sum(axis=1) == 0)] = (
        "Failure type unknown: Machine failure=1 with no mode flag"
    )
    reasons[n_predictable > 1] = "Multiple failure modes: no single class"

    excluded_mask = reasons != ""
    excluded_log = df.loc[excluded_mask, ["UDI", MACHINE_FAILURE_COLUMN] + PREDICTABLE_MODES + ["RNF"]].copy()
    excluded_log.insert(1, "reason", reasons[excluded_mask])

    clean = df.loc[~excluded_mask].copy()
    target = pd.Series(NO_FAILURE_LABEL, index=clean.index, dtype="object")
    for mode in PREDICTABLE_MODES:
        target[clean[mode] == 1] = mode
    clean[TARGET_COLUMN] = target

    _check_target_consistency(clean)
    return clean, excluded_log


def _check_target_consistency(clean: pd.DataFrame) -> None:
    """Fail loudly if the mapping produced something inconsistent."""
    failed_but_no_class = (clean[MACHINE_FAILURE_COLUMN] == 1) & (clean[TARGET_COLUMN] == NO_FAILURE_LABEL)
    class_but_not_failed = (clean[MACHINE_FAILURE_COLUMN] == 0) & (clean[TARGET_COLUMN] != NO_FAILURE_LABEL)
    if failed_but_no_class.any() or class_but_not_failed.any():
        raise ValueError(
            "Target mapping is inconsistent with 'Machine failure': "
            f"{int(failed_but_no_class.sum())} failed rows labelled No Failure, "
            f"{int(class_but_not_failed.sum())} non-failed rows given a failure class."
        )
    unknown = set(clean[TARGET_COLUMN]) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unexpected target labels: {unknown}")


def encode_target(labels: pd.Series) -> np.ndarray:
    """Map class names to fixed integer codes (XGBoost needs 0..K-1 integers)."""
    unknown = set(labels) - set(CLASS_TO_CODE)
    if unknown:
        raise ValueError(f"Cannot encode unknown labels: {unknown}")
    return labels.map(CLASS_TO_CODE).to_numpy(dtype=int)


def decode_target(codes) -> list[str]:
    """Map integer codes back to class names for reports and the app."""
    return [CLASS_NAMES[int(code)] for code in codes]


# ------------------------------------------------------ validity / types ---

def check_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Count values outside physical limits, plus unknown product types."""
    rows = []
    for column, (low, high) in VALIDITY_RULES.items():
        invalid = (df[column] < low) | (df[column] > high)
        rows.append({"check": f"{column} outside [{low}, {high}]", "count": int(invalid.sum())})
    rows.append(
        {
            "check": "Process temperature <= air temperature",
            "count": int((df["Process temperature [K]"] <= df["Air temperature [K]"]).sum()),
        }
    )
    rows.append(
        {
            "check": f"Type not in {PRODUCT_TYPES}",
            "count": int((~df["Type"].isin(PRODUCT_TYPES)).sum()),
        }
    )
    return pd.DataFrame(rows)


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Give model inputs explicit, stable dtypes (int sensors become float)."""
    out = df.copy()
    out[NUMERICAL_FEATURES] = out[NUMERICAL_FEATURES].astype("float64")
    out["Type"] = out["Type"].astype("object")
    return out


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """IQR outlier counts, and how often those rows are failures.

    Used to decide whether outliers are errors or real operating conditions;
    nothing is removed here.
    """
    overall_rate = df[MACHINE_FAILURE_COLUMN].mean()
    rows = []
    for column in NUMERICAL_FEATURES:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[column] < low) | (df[column] > high)
        rows.append(
            {
                "feature": column,
                "iqr_lower": round(low, 2),
                "iqr_upper": round(high, 2),
                "n_outliers": int(mask.sum()),
                "failure_rate_in_outliers_%": round(100 * df.loc[mask, MACHINE_FAILURE_COLUMN].mean(), 1)
                if mask.any()
                else None,
                "overall_failure_rate_%": round(100 * overall_rate, 2),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- features ---

def split_features_target(clean: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep only model inputs; identifiers and label columns never reach X."""
    features = clean[MODEL_INPUT_COLUMNS].copy()
    target = encode_target(clean[TARGET_COLUMN])
    return features, target


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Unfitted transformer for the model inputs.

    * Type: one-hot encoded with a fixed category list, so the output columns
      are identical in training, CV folds and the Streamlit app. handle_unknown
      is 'error' because a type other than L/M/H is invalid input, not
      something to encode as all zeros.
    * Numerical features: every numeric column that reaches this step (the
      original sensors plus any engineered features), selected by dtype so the
      same transformer works for every feature set compared in Stage 5.
      Passed through for tree models (splits are invariant to monotonic
      scaling); standardised only for scale-sensitive models such as Logistic
      Regression (scale_numeric=True).
    """
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "type",
                OneHotEncoder(categories=[PRODUCT_TYPES], handle_unknown="error", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", numeric_step, make_column_selector(dtype_include=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_train_test_split(clean: pd.DataFrame):
    """Stage 6: stratified 80/20 split, fixed seed.

    Returns X_train, X_test, y_train, y_test plus the UDIs of each part so the
    split can be saved and audited.
    """
    features, target = split_features_target(clean)
    return train_test_split(
        features,
        target,
        clean["UDI"].to_numpy(),
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def prepare_model_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full row-level cleaning used by every later stage.

    Returns the cleaned frame (inputs + target) and the exclusion log.
    """
    invalid = check_invalid_values(raw)
    if invalid["count"].sum() > 0:
        raise ValueError(f"Invalid values found:\n{invalid[invalid['count'] > 0]}")
    clean, excluded = build_target(raw)
    clean = enforce_dtypes(clean)
    return clean[["UDI"] + MODEL_INPUT_COLUMNS + [TARGET_COLUMN]], excluded


def main() -> None:
    from src.data_loading import load_raw_data
    from src.utils import ensure_dirs

    ensure_dirs()
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data()

    print("Missing values in raw data:", int(raw.isna().sum().sum()))
    print("Duplicate rows in raw data:", int(raw.duplicated().sum()))

    invalid = check_invalid_values(raw)
    invalid.to_csv(STAGE3_DIR / "invalid_value_checks.csv", index=False)
    print("\nInvalid-value checks:")
    print(invalid.to_string(index=False))

    outliers = outlier_summary(raw)
    outliers.to_csv(STAGE3_DIR / "outlier_summary.csv", index=False)
    print("\nIQR outlier investigation (nothing removed):")
    print(outliers.to_string(index=False))

    clean, excluded = prepare_model_data(raw)
    excluded.to_csv(STAGE3_DIR / "excluded_rows.csv", index=False)
    clean.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\nRows in: {len(raw)} | excluded: {len(excluded)} | kept: {len(clean)}")
    print("\nExclusions by reason:")
    print(excluded["reason"].value_counts().to_string())

    distribution = clean[TARGET_COLUMN].value_counts().reindex(CLASS_NAMES).rename("count").to_frame()
    distribution["percent"] = (100 * distribution["count"] / len(clean)).round(2)
    distribution["code"] = [CLASS_TO_CODE[name] for name in distribution.index]
    distribution.to_csv(STAGE3_DIR / "target_distribution.csv")
    print("\nTarget distribution:")
    print(distribution.to_string())

    with open(STAGE3_DIR / "class_mapping.json", "w", encoding="utf-8") as handle:
        json.dump(CLASS_TO_CODE, handle, indent=2)

    features, target = split_features_target(clean)
    print("\nModel input columns:", list(features.columns))
    print("Input dtypes:", dict(features.dtypes.astype(str)))
    print("Encoded target codes:", dict(zip(*np.unique(target, return_counts=True))))

    # Output-shape check only, on 5 rows; nothing fitted here is kept. (With a
    # fixed category list and passthrough numerics this transformer learns no
    # statistics anyway; the scaled variant is only ever fitted inside CV.)
    demo = build_preprocessor().fit(features.iloc[:5])
    print("Transformed feature names:", list(demo.get_feature_names_out()))
    print(f"\nSaved {CLEAN_DATA_PATH.relative_to(CLEAN_DATA_PATH.parents[2])} and reports/stage3/.")


if __name__ == "__main__":
    main()
