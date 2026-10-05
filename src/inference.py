"""Inference helpers shared by the Streamlit app and the tests.

The app never re-implements preprocessing: it builds a one-row DataFrame
with the raw input columns and passes it to the saved pipeline, which runs
the same FeatureEngineer -> ColumnTransformer -> model steps as training.
"""

import json
import math
from dataclasses import dataclass, field

import joblib
import pandas as pd

from config import (
    CLASS_NAMES,
    FINAL_MODEL_PATH,
    METADATA_PATH,
    MODEL_INPUT_COLUMNS,
    NUMERICAL_FEATURES,
    PRODUCT_TYPES,
)
from src.preprocessing import VALIDITY_RULES


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_artifacts(model_path=FINAL_MODEL_PATH, metadata_path=METADATA_PATH):
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run the training pipeline first (see README: 'How to train')."
        )
    pipeline = joblib.load(model_path)
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    return pipeline, metadata


def validate_inputs(inputs: dict, training_ranges: dict | None = None) -> ValidationResult:
    """Hard errors for missing/invalid values; warnings for extrapolation."""
    result = ValidationResult()
    for column in MODEL_INPUT_COLUMNS:
        if column not in inputs or inputs[column] is None or inputs[column] == "":
            result.errors.append(f"'{column}' is missing.")
    if result.errors:
        return result

    if inputs["Type"] not in PRODUCT_TYPES:
        result.errors.append(f"Type must be one of {PRODUCT_TYPES}, got '{inputs['Type']}'.")

    for column in NUMERICAL_FEATURES:
        value = inputs[column]
        try:
            number = float(value)
        except (TypeError, ValueError):
            result.errors.append(f"'{column}' must be a number, got '{value}'.")
            continue
        if not math.isfinite(number):
            result.errors.append(f"'{column}' must be a finite number.")
            continue
        low, high = VALIDITY_RULES[column]
        if not low <= number <= high:
            result.errors.append(f"'{column}' = {number} is physically implausible (allowed {low}–{high}).")
        elif training_ranges and not training_ranges[column][0] <= number <= training_ranges[column][1]:
            t_low, t_high = training_ranges[column]
            result.warnings.append(
                f"'{column}' = {number} is outside the training range {t_low}–{t_high}; "
                "the prediction is an extrapolation and less reliable."
            )

    if not result.errors and float(inputs["Process temperature [K]"]) <= float(inputs["Air temperature [K]"]):
        result.errors.append("Process temperature must be higher than air temperature.")
    return result


def build_input_frame(inputs: dict) -> pd.DataFrame:
    row = {"Type": str(inputs["Type"])}
    row.update({column: float(inputs[column]) for column in NUMERICAL_FEATURES})
    return pd.DataFrame([row], columns=MODEL_INPUT_COLUMNS)


def predict(pipeline, inputs: dict) -> tuple[str, pd.DataFrame]:
    """Return the predicted class name and a probability table (descending)."""
    frame = build_input_frame(inputs)
    code = int(pipeline.predict(frame)[0])
    proba = pipeline.predict_proba(frame)[0]
    table = pd.DataFrame({"Failure type": CLASS_NAMES, "Probability": proba})
    return CLASS_NAMES[code], table.sort_values("Probability", ascending=False).reset_index(drop=True)
