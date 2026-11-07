"""SHAP-based explainable AI helpers.

Loads the trained TreeExplainer and produces real per-prediction feature
contributions for a single patient. This powers the /explain endpoint and the
"Explainable AI" page.
"""
from __future__ import annotations

import joblib
import pandas as pd

from src.config import EXPLAINER_PATH, FEATURE_ORDER


def load_explainer() -> dict:
    """Load the saved SHAP explainer bundle."""
    if not EXPLAINER_PATH.exists():
        raise FileNotFoundError(
            "SHAP explainer not found. Train the models first (src/train.py)."
        )
    return joblib.load(EXPLAINER_PATH)


def build_input_frame(features: dict) -> pd.DataFrame:
    """Convert an API payload into a single-row DataFrame in training column order."""
    row = {col: features.get(col) for col in FEATURE_ORDER}
    return pd.DataFrame([row])


def explain_prediction(features: dict) -> dict:
    """Return SHAP explanation for a single patient's input.

    Returns feature contributions (logit / margin space), the expected value and
    the transformed feature names so the UI can render a waterfall/bar chart.
    """
    bundle = load_explainer()
    preprocessor = bundle["preprocessor"]
    explainer = bundle["explainer"]
    feature_names = bundle["feature_names"]

    frame = build_input_frame(features)
    x_transformed = preprocessor.transform(frame)
    shap_values = explainer.shap_values(x_transformed)

    # shap_values: (1, n_features) for binary classification -> flatten
    values = shap_values.ravel().tolist() if hasattr(shap_values, "ravel") else list(shap_values)

    return {
        "base_value": float(explainer.expected_value),
        "feature_names": feature_names,
        "shap_values": [float(v) for v in values],  # margin-space contributions
        "positive_class_label": "Heart disease present",
    }
