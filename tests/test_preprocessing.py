"""Test 2 - cleaning, target construction, preprocessing and the split."""

import numpy as np
import pandas as pd
import pytest

from config import CLASS_NAMES, MODEL_INPUT_COLUMNS, TARGET_COLUMN
from src.preprocessing import (
    build_preprocessor,
    build_target,
    decode_target,
    encode_target,
    make_train_test_split,
    prepare_model_data,
    split_features_target,
)


@pytest.fixture(scope="module")
def clean_and_excluded(raw_df):
    return prepare_model_data(raw_df)


def test_exclusions_are_logged_and_counted(clean_and_excluded, raw_df):
    clean, excluded = clean_and_excluded
    assert len(excluded) == 50
    assert len(clean) + len(excluded) == len(raw_df)
    assert excluded["reason"].value_counts().to_dict() == {
        "Multiple failure modes: no single class": 23,
        "Contradictory labels: RNF=1 but Machine failure=0": 18,
        "Failure type unknown: Machine failure=1 with no mode flag": 9,
    }
    assert not set(excluded["UDI"]) & set(clean["UDI"])


def test_class_distribution(clean_and_excluded):
    clean, _ = clean_and_excluded
    counts = clean[TARGET_COLUMN].value_counts().to_dict()
    assert counts == {"No Failure": 9643, "HDF": 106, "PWF": 80, "OSF": 78, "TWF": 43}


def test_target_consistent_with_machine_failure(raw_df):
    clean, _ = build_target(raw_df)
    failed = clean["Machine failure"] == 1
    assert (clean.loc[failed, TARGET_COLUMN] != "No Failure").all()
    assert (clean.loc[~failed, TARGET_COLUMN] == "No Failure").all()


def test_no_identifier_or_label_leaks_into_features(clean_and_excluded):
    X, _ = split_features_target(clean_and_excluded[0])
    assert list(X.columns) == MODEL_INPUT_COLUMNS
    for leaky in ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF", TARGET_COLUMN]:
        assert leaky not in X.columns


def test_encode_decode_round_trip():
    codes = encode_target(pd.Series(CLASS_NAMES))
    assert list(codes) == list(range(len(CLASS_NAMES)))
    assert decode_target(codes) == CLASS_NAMES
    with pytest.raises(ValueError):
        encode_target(pd.Series(["RNF"]))


def test_preprocessor_output_and_invalid_type(clean_and_excluded):
    X, _ = split_features_target(clean_and_excluded[0])
    pre = build_preprocessor().fit(X.head(20))
    out = pre.transform(X.head(20))
    assert out.shape == (20, 8)  # 3 one-hot Type columns + 5 numeric
    assert np.allclose(out[:, :3].sum(axis=1), 1)
    bad = X.head(1).copy()
    bad["Type"] = "Z"
    with pytest.raises(ValueError):
        pre.transform(bad)


def test_split_is_stratified_and_disjoint(clean_and_excluded):
    X_tr, X_te, y_tr, y_te, udi_tr, udi_te = make_train_test_split(clean_and_excluded[0])
    assert len(X_te) / (len(X_tr) + len(X_te)) == pytest.approx(0.2, abs=0.001)
    assert not set(udi_tr) & set(udi_te)
    for code in range(len(CLASS_NAMES)):
        assert (y_tr == code).mean() == pytest.approx((y_te == code).mean(), abs=0.002)


def test_split_is_reproducible(clean_and_excluded):
    first = make_train_test_split(clean_and_excluded[0])[5]
    second = make_train_test_split(clean_and_excluded[0])[5]
    assert np.array_equal(first, second)
