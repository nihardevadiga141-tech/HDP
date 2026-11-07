"""FastAPI backend for Heart Disease Prediction.

Endpoints:
  GET  /health      - service health + model availability
  GET  /model-info   - model names, features and real evaluation metrics
  POST /predict      - run prediction on clinical parameters
  POST /explain      - SHAP explanation for a patient
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.prediction import model_loaded, predict
from backend.schemas import (
    ExplainResponse,
    HealthResponse,
    HeartPredictRequest,
    ModelInfoResponse,
    PredictResponse,
)
from src.config import ENSEMBLE_PIPELINE, FEATURE_ORDER, HEART_CSV, MODEL_NAMES
from src.evaluation import load_metrics
from src.explainability import explain_prediction

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "ML-based cardiovascular risk screening. This service is a screening aid "
        "and does NOT provide medical diagnoses."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded(),
        model_path=str(ENSEMBLE_PIPELINE),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        models=MODEL_NAMES,
        features=FEATURE_ORDER,
        metrics=load_metrics(),
        source_dataset=str(HEART_CSV),
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: HeartPredictRequest) -> PredictResponse:
    try:
        result = predict(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    return PredictResponse(**result)


@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(payload: HeartPredictRequest) -> ExplainResponse:
    try:
        explanation = explain_prediction(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc
    return ExplainResponse(**explanation)
