"""Central configuration for the Heart Disease Prediction project.

All feature names, model settings and file paths live here so the rest of the
codebase stays consistent and easy to maintain.
"""
from __future__ import annotations

import pathlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
HEART_CSV = DATA_DIR / "heart.csv"

MODELS_DIR = PROJECT_ROOT / "models"
ENSEMBLE_PIPELINE = MODELS_DIR / "ensemble_pipeline.joblib"
INDIVIDUAL_MODELS = MODELS_DIR / "individual_models.joblib"
METRICS_JSON = MODELS_DIR / "metrics.json"
FEATURE_IMPORTANCES_JSON = MODELS_DIR / "feature_importances.json"
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"

PLOTS_DIR = PROJECT_ROOT / "models" / "plots"

# ---------------------------------------------------------------------------
# Data / feature configuration
# ---------------------------------------------------------------------------
TARGET = "target"

# Continuous features that are scaled with StandardScaler.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Nominal categorical features that are one-hot encoded.
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]

# Already binary 0/1 features, passed through as-is.
BINARY_FEATURES = ["sex", "fbs", "exang"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
FEATURE_ORDER = ALL_FEATURES

# ---------------------------------------------------------------------------
# Modelling configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Ensemble (Soft Voting)"]
