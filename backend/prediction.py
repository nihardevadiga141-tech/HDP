"""Prediction engine: loads the saved ensemble pipeline and runs inference."""
from __future__ import annotations

import joblib
import pandas as pd

from src.config import ENSEMBLE_PIPELINE, FEATURE_ORDER
from utils.recommendations import get_recommendations

_model = None


def _load_model():
    global _model
    if _model is None:
        if not ENSEMBLE_PIPELINE.exists():
            raise FileNotFoundError(
                "Trained pipeline not found. Train the model first (python -m src.train)."
            )
        _model = joblib.load(ENSEMBLE_PIPELINE)
    return _model


def model_loaded() -> bool:
    try:
        _load_model()
        return True
    except Exception:
        return False


def predict(features: dict) -> dict:
    """Return prediction, probability and recommendations for a patient."""
    model = _load_model()

    row = {col: features.get(col) for col in FEATURE_ORDER}
    frame = pd.DataFrame([row])

    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(model.predict(frame)[0])

    recommendations = get_recommendations(probability, features)

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_category": recommendations["risk_category"],
        "recommendations": recommendations,
        "disclaimer": recommendations["disclaimer"],
        "model": "Ensemble (Soft Voting: Logistic Regression + Random Forest + XGBoost)",
    }
