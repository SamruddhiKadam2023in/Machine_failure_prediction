"""Load the raw AI4I 2020 dataset and validate its schema."""

from pathlib import Path

import pandas as pd

from config import EXPECTED_COLUMNS, RAW_DATA_PATH


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV without modifying it on disk.

    The UCI file starts with a UTF-8 byte-order mark, so 'utf-8-sig' is used;
    otherwise the first column would be read as '\\ufeffUDI'.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place ai4i2020.csv in data/raw/."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df
