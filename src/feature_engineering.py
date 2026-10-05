"""Stage 5 - Feature engineering.

FeatureEngineer is an sklearn transformer, so it sits inside the model
pipeline and runs identically in training, cross-validation and the
Streamlit app. It is stateless (it learns nothing from data), so it cannot
leak information from validation or test rows.

Engineered features (each is a physical quantity, not a random combination):

* Temp difference [K] = Process temperature - Air temperature
    Heat leaves the process through the air. A small gap means poor heat
    dissipation; EDA shows HDF clustering at the smallest gaps and low rpm.

* Power [W] = Torque [Nm] x Rotational speed [rpm] x 2*pi/60
    Mechanical power delivered by the spindle. EDA shows PWF at both ends of
    the speed-torque curve (high torque/low rpm and low torque/high rpm),
    which a tree needs two separate splits to express in raw features but
    one threshold pair on power.

* Strain [min*Nm] = Tool wear [min] x Torque [Nm]
    Load on an already-worn tool. EDA shows OSF only where both tool wear and
    torque are high, i.e. at the top of this product.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import MODEL_INPUT_COLUMNS

ENGINEERED_FEATURES = {
    "Temp difference [K]": "Process temperature [K] - Air temperature [K]",
    "Power [W]": "Torque [Nm] * Rotational speed [rpm] * 2*pi/60",
    "Strain [min*Nm]": "Tool wear [min] * Torque [Nm]",
}

# Candidate feature sets compared with CV on the training split (Stage 5).
FEATURE_SETS = {
    "original": {"add_engineered": False, "drop_columns": ()},
    "original + engineered": {"add_engineered": True, "drop_columns": ()},
    # Air and process temperature are 0.88-correlated and their physically
    # relevant information for HDF is the difference, so this set tests
    # whether the two raw temperatures are redundant once the gap is present.
    "engineered, raw temperatures dropped": {
        "add_engineered": True,
        "drop_columns": ("Air temperature [K]", "Process temperature [K]"),
    },
}


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Temp difference [K]"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    out["Power [W]"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"] * 2 * np.pi / 60
    out["Strain [min*Nm]"] = out["Tool wear [min]"] * out["Torque [Nm]"]
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add physics-based features and optionally drop redundant raw columns."""

    def __init__(self, add_engineered: bool = True, drop_columns: tuple = ()):
        self.add_engineered = add_engineered
        self.drop_columns = drop_columns

    def fit(self, X, y=None):
        self._check_input(X)
        self.feature_names_in_ = np.array(MODEL_INPUT_COLUMNS, dtype=object)
        self.n_features_in_ = len(MODEL_INPUT_COLUMNS)
        return self

    def transform(self, X):
        self._check_input(X)
        out = X[MODEL_INPUT_COLUMNS].copy()
        if self.add_engineered:
            out = add_engineered_features(out)
        return out.drop(columns=list(self.drop_columns))

    def get_feature_names_out(self, input_features=None):
        names = list(MODEL_INPUT_COLUMNS)
        if self.add_engineered:
            names += list(ENGINEERED_FEATURES)
        return np.array([n for n in names if n not in self.drop_columns], dtype=object)

    @staticmethod
    def _check_input(X) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame with named columns.")
        missing = [c for c in MODEL_INPUT_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
