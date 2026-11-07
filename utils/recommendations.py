"""General preventive suggestion generator.

Suggestions are derived from the *predicted probability / risk category* and
the user's own clinical inputs. These are awareness messages, not medical
advice, and the app always directs high-risk users to a professional.
"""
from __future__ import annotations


RISK_LOW = "Low"
RISK_MODERATE = "Moderate"
RISK_HIGH = "High"


def classify_risk(probability: float) -> str:
    """Map a predicted probability to a human-readable risk category."""
    if probability < 0.40:
        return RISK_LOW
    if probability <= 0.60:
        return RISK_MODERATE
    return RISK_HIGH


def get_recommendations(probability: float, features: dict) -> dict:
    """Return structured, general preventive suggestions for a given risk level."""
    category = classify_risk(probability)

    general = {
        RISK_LOW: [
            "Your predicted risk is low — keep up your current healthy habits.",
            "Maintain a balanced diet with plenty of fruit, vegetables and whole grains.",
            "Aim for at least 150 minutes of moderate physical activity per week.",
            "Keep routine health check-ups to monitor blood pressure and cholesterol.",
        ],
        RISK_MODERATE: [
            "Your predicted risk is moderate — it is worth paying closer attention to heart health.",
            "Review your diet: reduce salt, saturated fat and processed foods.",
            "Add regular moderate exercise and manage stress levels.",
            "Consider discussing your numbers with a healthcare professional.",
        ],
        RISK_HIGH: [
            "Your predicted risk is elevated. Please do not ignore this result.",
            "We strongly encourage you to consult a doctor or cardiologist soon.",
            "An ECG, blood test (lipid profile) and blood pressure check would be useful next steps.",
            "Avoid smoking, limit alcohol, and keep blood pressure/cholesterol in check.",
        ],
    }

    specific: list[str] = []
    if features.get("chol", 0) >= 220:
        specific.append("Your cholesterol reading is high — a lipid profile review is advisable.")
    if features.get("trestbps", 0) >= 130:
        specific.append("Your resting blood pressure is elevated — monitor it regularly.")
    if features.get("thalach", 999) < 100:
        specific.append("Your achieved maximum heart rate is on the lower side; discuss with a clinician.")
    if features.get("oldpeak", 0) > 1.5:
        specific.append("A notable ST-depression value was reported — this warrants clinical review.")
    if features.get("fbs", 0) == 1:
        specific.append("Fasting blood sugar above 120 mg/dL was flagged — a glucose test is recommended.")

    return {
        "risk_category": category,
        "general": general[category],
        "specific": specific,
        "disclaimer": (
            "This is a screening aid based on a machine-learning model. It is NOT a medical "
            "diagnosis. Always consult a qualified healthcare professional for clinical decisions."
        ),
    }
