"""Streamlit app: predict the machine failure type from operating parameters.

Run from the project root:

    streamlit run app.py

The app loads models/final_model.joblib, which contains the complete fitted
pipeline (feature engineering -> preprocessing -> model). It only collects
raw inputs; all transformations happen inside that saved pipeline, exactly
as during training.
"""

import streamlit as st

from config import CLASS_NAMES, NUMERICAL_FEATURES, PRODUCT_TYPES
from src.feature_engineering import ENGINEERED_FEATURES, add_engineered_features
from src.inference import build_input_frame, load_artifacts, predict, validate_inputs

st.set_page_config(page_title="Machine Failure-Type Prediction", page_icon="⚙️", layout="centered")

TYPE_LABELS = {"L": "L — low quality variant", "M": "M — medium quality variant", "H": "H — high quality variant"}
INPUT_HELP = {
    "Air temperature [K]": "Ambient air temperature in kelvin.",
    "Process temperature [K]": "Temperature of the machining process in kelvin (must exceed air temperature).",
    "Rotational speed [rpm]": "Spindle speed in revolutions per minute.",
    "Torque [Nm]": "Torque in newton-metres.",
    "Tool wear [min]": "Minutes the current tool has been in use.",
}
STEPS = {
    "Air temperature [K]": 0.1,
    "Process temperature [K]": 0.1,
    "Rotational speed [rpm]": 1.0,
    "Torque [Nm]": 0.1,
    "Tool wear [min]": 1.0,
}


@st.cache_resource(show_spinner="Loading model…")
def get_artifacts():
    return load_artifacts()


def render_model_info(metadata: dict) -> None:
    with st.sidebar:
        st.header("Model information")
        st.markdown(f"**Model:** {metadata['model_name']}")
        st.markdown(f"**Selected by:** {metadata['selection_rule']}")
        test = metadata["test_metrics"]
        st.markdown("**Held-out test set**")
        st.markdown(
            f"- Macro F1: {test['macro_f1']:.3f}\n"
            f"- Accuracy: {test['accuracy']:.3f}\n"
            f"- Macro ROC-AUC (OvR): {test['macro_roc_auc']:.3f}"
        )
        st.markdown(
            f"**CV Macro F1:** {metadata['cv_macro_f1_mean']:.3f} ± {metadata['cv_macro_f1_std']:.3f}"
        )
        st.markdown("**Failure types**")
        for code in CLASS_NAMES:
            st.markdown(f"- **{code}** — {metadata['class_descriptions'][code]}")
        st.caption(
            "Trained on the UCI AI4I 2020 synthetic predictive-maintenance dataset. "
            "Predictions support, but do not replace, engineering judgement."
        )


def main() -> None:
    st.title("⚙️ Machine Failure-Type Prediction")
    st.write(
        "Enter the machine's current operating parameters to predict which failure mode, if any, "
        "the operating point corresponds to: tool wear, heat dissipation, power, or overstrain failure."
    )

    try:
        pipeline, metadata = get_artifacts()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()
    except Exception as error:  # corrupted file, version mismatch, etc.
        st.error(f"The model could not be loaded: {error}")
        st.stop()

    render_model_info(metadata)
    medians = metadata["training_medians"]
    ranges = metadata["training_ranges"]

    with st.form("inputs"):
        st.subheader("Operating parameters")
        product_type = st.selectbox(
            "Product quality variant (Type)", PRODUCT_TYPES, format_func=TYPE_LABELS.get,
            help="Quality variant of the product being machined.",
        )
        left, right = st.columns(2)
        values = {}
        for i, column in enumerate(NUMERICAL_FEATURES):
            target = left if i % 2 == 0 else right
            low, high = ranges[column]
            values[column] = target.number_input(
                column, value=float(medians[column]), step=STEPS[column], format="%.1f",
                help=f"{INPUT_HELP[column]} Training range: {low:g}–{high:g}.",
            )
        submitted = st.form_submit_button("Predict failure type", type="primary", width="stretch")

    if not submitted:
        st.info("Adjust the parameters and press **Predict failure type**.")
        return

    inputs = {"Type": product_type, **values}
    check = validate_inputs(inputs, training_ranges=ranges)
    for message in check.errors:
        st.error(message)
    if not check.ok:
        return
    for message in check.warnings:
        st.warning(message)

    try:
        label, probabilities = predict(pipeline, inputs)
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        return

    description = metadata["class_descriptions"][label]
    confidence = float(probabilities.loc[probabilities["Failure type"] == label, "Probability"].iloc[0])
    st.subheader("Prediction")
    if label == "No Failure":
        st.success(f"**Predicted failure type: {label}** — confidence {confidence:.2%}")
    else:
        st.error(f"**Predicted failure type: {label} ({description})** — confidence {confidence:.2%}")

    st.markdown("**Probability for each failure type**")
    chart_data = probabilities.set_index("Failure type").reindex(CLASS_NAMES)
    st.bar_chart(chart_data, y="Probability", horizontal=True)
    table = probabilities.assign(
        Description=probabilities["Failure type"].map(metadata["class_descriptions"]),
        Probability=probabilities["Probability"].map(lambda p: f"{p:.2%}"),
    )[["Failure type", "Description", "Probability"]]
    st.dataframe(table, hide_index=True, width="stretch")

    with st.expander("Derived quantities computed by the model pipeline"):
        engineered = add_engineered_features(build_input_frame(inputs))[list(ENGINEERED_FEATURES)]
        derived = engineered.T.reset_index()
        derived.columns = ["Quantity", "Value"]
        st.dataframe(derived.round(2), hide_index=True, width="stretch")
        st.caption("Computed with the same feature-engineering code the model uses; shown for information.")


main()
