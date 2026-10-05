"""Test 1 - dataset loading."""

import pandas as pd
import pytest

from config import EXPECTED_COLUMNS
from src.data_loading import load_raw_data


def test_raw_data_shape_and_columns(raw_df):
    assert raw_df.shape == (10000, 14)
    assert list(raw_df.columns) == EXPECTED_COLUMNS  # BOM stripped: first column is 'UDI'


def test_raw_data_has_no_missing_values(raw_df):
    assert int(raw_df.isna().sum().sum()) == 0


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "does_not_exist.csv")


def test_missing_column_raises(tmp_path, raw_df):
    path = tmp_path / "broken.csv"
    raw_df.drop(columns=["Torque [Nm]"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        load_raw_data(path)


def test_loading_does_not_modify_raw_file(raw_df):
    assert load_raw_data().equals(raw_df)
    assert isinstance(raw_df, pd.DataFrame)
