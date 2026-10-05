"""Test 3 - feature engineering."""

import math

import pandas as pd
import pytest

from src.feature_engineering import ENGINEERED_FEATURES, FeatureEngineer, add_engineered_features


@pytest.fixture
def one_row(valid_inputs):
    return pd.DataFrame([valid_inputs])


def test_engineered_formulas(one_row):
    out = add_engineered_features(one_row).iloc[0]
    assert out["Temp difference [K]"] == pytest.approx(10.0)
    assert out["Power [W]"] == pytest.approx(40.0 * 1500.0 * 2 * math.pi / 60)
    assert out["Strain [min*Nm]"] == pytest.approx(100.0 * 40.0)


def test_transformer_adds_expected_columns(one_row):
    out = FeatureEngineer(add_engineered=True).fit(one_row).transform(one_row)
    for name in ENGINEERED_FEATURES:
        assert name in out.columns
    assert list(out.columns) == list(FeatureEngineer().get_feature_names_out())


def test_drop_columns_and_original_only(one_row):
    dropped = FeatureEngineer(drop_columns=("Air temperature [K]",)).fit_transform(one_row)
    assert "Air temperature [K]" not in dropped.columns
    original = FeatureEngineer(add_engineered=False).fit_transform(one_row)
    assert not set(ENGINEERED_FEATURES) & set(original.columns)


def test_transformer_is_stateless(one_row, raw_df):
    """Fitting on different data must not change the transform (no leakage)."""
    fe_a = FeatureEngineer().fit(one_row)
    fe_b = FeatureEngineer().fit(raw_df)
    pd.testing.assert_frame_equal(fe_a.transform(one_row), fe_b.transform(one_row))


def test_missing_column_is_rejected(one_row):
    with pytest.raises(ValueError, match="missing required columns"):
        FeatureEngineer().fit(one_row).transform(one_row.drop(columns=["Torque [Nm]"]))


def test_non_dataframe_is_rejected(one_row):
    with pytest.raises(TypeError):
        FeatureEngineer().fit(one_row).transform(one_row.to_numpy())
