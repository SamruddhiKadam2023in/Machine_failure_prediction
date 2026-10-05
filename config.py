"""Project-wide configuration: paths, column groups and the random seed.

Every other module imports from here so paths and column names are defined once.
All paths are relative to the project root, never absolute.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ai4i2020.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ID_COLUMNS = ["UDI", "Product ID"]
CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
MACHINE_FAILURE_COLUMN = "Machine failure"
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Target definition approved in Stage 2 (see reports/01_02_...md, section 2.8.1).
# RNF is not a class: it is random and independent of the inputs.
TARGET_COLUMN = "Failure Type"
PREDICTABLE_MODES = ["TWF", "HDF", "PWF", "OSF"]
NO_FAILURE_LABEL = "No Failure"
# Fixed order -> fixed integer codes, independent of alphabetical sorting.
CLASS_NAMES = [NO_FAILURE_LABEL] + PREDICTABLE_MODES
CLASS_TO_CODE = {name: code for code, name in enumerate(CLASS_NAMES)}
CLASS_DESCRIPTIONS = {
    "No Failure": "No failure",
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
}
PRODUCT_TYPES = ["L", "M", "H"]
MODEL_INPUT_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

EXPECTED_COLUMNS = (
    ID_COLUMNS
    + CATEGORICAL_FEATURES
    + NUMERICAL_FEATURES
    + [MACHINE_FAILURE_COLUMN]
    + FAILURE_MODE_COLUMNS
)
