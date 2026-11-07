"""Pydantic request/response models for the FastAPI backend."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HeartPredictRequest(BaseModel):
    """Clinical parameters for a single prediction (all fields required)."""

    age: int = Field(ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(description="0 = female, 1 = male")
    cp: Literal[0, 1, 2, 3] = Field(description="Chest pain type")
    trestbps: int = Field(ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(description="Resting ECG results")
    thalach: int = Field(ge=40, le=260, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(description="Exercise-induced angina")
    oldpeak: float = Field(ge=0.0, le=7.0, description="ST depression")
    slope: Literal[0, 1, 2] = Field(description="Slope of peak exercise ST segment")
    ca: Literal[0, 1, 2, 3, 4] = Field(description="Major vessels colored by fluoroscopy")
    thal: Literal[1, 2, 3] = Field(description="Thalassemia type (1=normal, 2=fixed defect, 3=reversible defect)")


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    risk_category: str
    recommendations: dict
    disclaimer: str
    model: str


class ExplainResponse(BaseModel):
    base_value: float
    feature_names: list[str]
    shap_values: list[float]
    positive_class_label: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    models: list[str]
    features: list[str]
    metrics: dict
    source_dataset: str
