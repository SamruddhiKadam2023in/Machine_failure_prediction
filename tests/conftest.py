"""Shared fixtures. Tests run from the project root: `pytest`."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FINAL_MODEL_PATH, METADATA_PATH  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    from src.data_loading import load_raw_data

    return load_raw_data()


@pytest.fixture(scope="session")
def artifacts():
    if not FINAL_MODEL_PATH.exists() or not METADATA_PATH.exists():
        pytest.skip("Final model not trained yet; run the pipeline first.")
    from src.inference import load_artifacts

    return load_artifacts()


@pytest.fixture
def valid_inputs():
    return {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }
