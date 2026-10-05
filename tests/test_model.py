"""Tests 4-6 - model loading, prediction, valid output classes."""

import joblib
import numpy as np
import pandas as pd

from config import CLASS_NAMES, MODEL_INPUT_COLUMNS, PREPROCESSING_PATH
from src.feature_study import load_split
from src.inference import build_input_frame, predict


def test_final_model_loads(artifacts):
    pipeline, metadata = artifacts
    assert list(pipeline.named_steps) == ["features", "preprocess", "model"]
    assert metadata["class_names"] == CLASS_NAMES
    assert metadata["input_columns"] == MODEL_INPUT_COLUMNS


def test_prediction_is_a_valid_class(artifacts, valid_inputs):
    pipeline, _ = artifacts
    label, probabilities = predict(pipeline, valid_inputs)
    assert label in CLASS_NAMES
    assert set(probabilities["Failure type"]) == set(CLASS_NAMES)
    assert abs(probabilities["Probability"].sum() - 1.0) < 1e-6
    assert probabilities.iloc[0]["Failure type"] == label  # highest probability is the prediction


def test_batch_predictions_are_valid_codes(artifacts):
    pipeline, _ = artifacts
    _, X_test, _, _ = load_split()
    codes = pipeline.predict(X_test)
    assert codes.shape == (len(X_test),)
    assert set(np.unique(codes)) <= set(range(len(CLASS_NAMES)))
    proba = pipeline.predict_proba(X_test)
    assert proba.shape == (len(X_test), len(CLASS_NAMES))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_saved_metrics_reproduce(artifacts):
    """The test score recorded in the metadata is reproducible from the saved model."""
    from sklearn.metrics import f1_score

    pipeline, metadata = artifacts
    _, X_test, _, y_test = load_split()
    score = f1_score(y_test, pipeline.predict(X_test), average="macro")
    assert abs(score - metadata["test_metrics"]["macro_f1"]) < 1e-9


def test_preprocessing_pipeline_matches_final_model(artifacts, valid_inputs):
    pipeline, _ = artifacts
    preprocessing = joblib.load(PREPROCESSING_PATH)
    frame = build_input_frame(valid_inputs)
    expected = pipeline.named_steps["preprocess"].transform(pipeline.named_steps["features"].transform(frame))
    assert np.allclose(preprocessing.transform(frame), expected)


def test_known_failure_patterns(artifacts):
    """Sanity check against unmistakable training patterns (not a metric)."""
    pipeline, _ = artifacts
    X_train, _, y_train, _ = load_split()
    # Predictions on the training rows of each failure class should mostly
    # recover that class; a broken pipeline would collapse to 'No Failure'.
    for code, name in enumerate(CLASS_NAMES[1:], start=1):
        rows = X_train[y_train == code]
        assert (pipeline.predict(rows) == code).mean() > 0.5, name
    assert isinstance(X_train, pd.DataFrame)
