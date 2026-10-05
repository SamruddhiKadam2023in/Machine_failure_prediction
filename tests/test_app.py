"""Test 7 - Streamlit input compatibility and the running app."""

from pathlib import Path

import pytest

from config import MODEL_INPUT_COLUMNS
from src.inference import build_input_frame, validate_inputs

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_valid_inputs_pass(valid_inputs):
    result = validate_inputs(valid_inputs)
    assert result.ok and not result.warnings


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("Type", "X", "Type must be one of"),
        ("Torque [Nm]", "abc", "must be a number"),
        ("Torque [Nm]", float("nan"), "finite"),
        ("Rotational speed [rpm]", -5, "physically implausible"),
        ("Tool wear [min]", None, "missing"),
    ],
)
def test_invalid_inputs_are_rejected(valid_inputs, column, value, message):
    valid_inputs[column] = value
    result = validate_inputs(valid_inputs)
    assert not result.ok
    assert any(message in error for error in result.errors)


def test_process_must_exceed_air_temperature(valid_inputs):
    valid_inputs["Process temperature [K]"] = valid_inputs["Air temperature [K]"]
    assert not validate_inputs(valid_inputs).ok


def test_out_of_training_range_warns(valid_inputs):
    ranges = {c: [0.0, 1e9] for c in MODEL_INPUT_COLUMNS[1:]}
    ranges["Torque [Nm]"] = [3.8, 76.6]
    valid_inputs["Torque [Nm]"] = 90.0
    result = validate_inputs(valid_inputs, training_ranges=ranges)
    assert result.ok and any("outside the training range" in w for w in result.warnings)


def test_input_frame_matches_training_schema(valid_inputs):
    frame = build_input_frame(valid_inputs)
    assert list(frame.columns) == MODEL_INPUT_COLUMNS
    assert frame.shape == (1, len(MODEL_INPUT_COLUMNS))
    assert frame["Type"].iloc[0] == "M"
    assert all(frame[c].dtype == "float64" for c in MODEL_INPUT_COLUMNS[1:])


def test_streamlit_app_predicts(artifacts):
    """Run the real app headlessly, submit the form, and check the result."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown) + " ".join(
        e.value for e in list(app.success) + list(app.error)
    )
    assert "Predicted failure type" in text


def test_streamlit_app_rejects_invalid_temperatures(artifacts):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    process = next(w for w in app.number_input if w.label == "Process temperature [K]")
    process.set_value(290.0)  # below the default air temperature
    app.button[0].click().run()
    assert any("Process temperature must be higher" in e.value for e in app.error)
    assert not any("Predicted failure type" in e.value for e in list(app.success) + list(app.error))
