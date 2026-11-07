"""End-to-end training script.

Runs the full ML workflow:
  1. load + clean data
  2. stratified train/test split
  3. build the shared preprocessor
  4. train Logistic Regression, Random Forest, XGBoost
  5. build a soft-voting ensemble
  6. evaluate all on the test set (accuracy/precision/recall/F1/ROC-AUC/CM)
  7. save pipelines (joblib), metrics + ROC curves (JSON/PNG), feature
     importances, and a SHAP explainer.

Every number written to disk is computed from real test predictions.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import (
    ENSEMBLE_PIPELINE,
    EXPLAINER_PATH,
    FEATURE_IMPORTANCES_JSON,
    INDIVIDUAL_MODELS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.evaluation import (
    evaluate_classifier,
    roc_points,
    save_confusion_matrices,
    save_roc_curves,
    write_metrics,
)
from src.preprocessing import build_preprocessor, clean_data, load_data, prepare_features_target


def build_models(preprocessor):
    """Return {name: pipeline} for the three individual models."""
    lr_pipeline = _pipeline(preprocessor, "lr", LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=RANDOM_STATE
    ))
    rf_pipeline = _pipeline(preprocessor, "rf", RandomForestClassifier(
        n_estimators=300, min_samples_split=5, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    ))
    xgb_pipeline = _pipeline(preprocessor, "xgb", XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_STATE,
    ))
    return {
        "Logistic Regression": lr_pipeline,
        "Random Forest": rf_pipeline,
        "XGBoost": xgb_pipeline,
    }


def _pipeline(preprocessor, name, estimator):
    from sklearn.pipeline import Pipeline

    return Pipeline(steps=[("prep", preprocessor), (name, estimator)])


def build_ensemble(individual_models):
    """Soft-voting ensemble of the three individual pipelines."""
    return VotingClassifier(
        estimators=[(key.lower().replace(" ", "_"), model) for key, model in individual_models.items()],
        voting="soft",
        n_jobs=-1,
    )


def train_and_evaluate() -> dict:
    """Run the whole workflow and return summary info."""
    raw = load_data()
    cleaned = clean_data(raw)
    print(f"Clean data shape: {cleaned.shape}")

    X, y = prepare_features_target(cleaned)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    preprocessor = build_preprocessor()

    individual = build_models(preprocessor)
    ensemble = build_ensemble(individual)

    # Fit everything
    for name, model in individual.items():
        model.fit(X_train, y_train)
        print(f"Fitted {name}")
    ensemble.fit(X_train, y_train)
    print("Fitted Ensemble (soft voting)")

    # Evaluate on the held-out test set
    metrics = {}
    roc_data = {}
    cm_data = {}
    for name, model in {**individual, "Ensemble (Soft Voting)": ensemble}.items():
        metrics[name] = evaluate_classifier(model, X_test, y_test)
        pts = roc_points(model, X_test, y_test)
        pts["roc_auc"] = metrics[name]["roc_auc"]
        roc_data[name] = pts
        cm_data[name] = metrics[name]["confusion_matrix"]
        m = metrics[name]
        print(
            f"{name:26s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
            f"rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f}"
        )

    write_metrics(metrics, roc_data)
    save_roc_curves(roc_data)
    save_confusion_matrices(cm_data)

    # Persist pipelines
    ENSEMBLE_PIPELINE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ensemble, ENSEMBLE_PIPELINE)
    joblib.dump(individual, INDIVIDUAL_MODELS)

    # Feature importances from the tree models
    feature_names = preprocessor.named_steps["prep"].get_feature_names_out()
    _write_feature_importances(individual, feature_names)

    # SHAP explainer (tree-based, works on the transformed feature space)
    _build_shap_explainer(preprocessor, X_train, y_train)

    return {
        "X_test": X_test,
        "y_test": y_test,
        "metrics": metrics,
        "roc_data": roc_data,
    }


def _write_feature_importances(individual: dict, feature_names) -> None:
    payload = {}
    for name in ["Random Forest", "XGBoost"]:
        if name in individual:
            clf = individual[name].named_steps[list(individual[name].named_steps)[-1]]
            importance = [float(v) for v in clf.feature_importances_]
            payload[name] = {
                "feature_names": [str(f) for f in feature_names],
                "importance": importance,
            }
    FEATURE_IMPORTANCES_JSON.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_IMPORTANCES_JSON.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _build_shap_explainer(preprocessor, X_train, y_train) -> None:
    """Fit a SHAP TreeExplainer on an XGBoost model trained on transformed features."""
    import shap

    X_trans = preprocessor.named_steps["prep"].fit_transform(X_train)
    feature_names = [str(f) for f in preprocessor.named_steps["prep"].get_feature_names_out()]

    model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
    model.fit(X_trans, y_train.values.ravel())
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    EXPLAINER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "explainer": explainer,
            "preprocessor": preprocessor.named_steps["prep"],
            "feature_names": feature_names,
            "model": model,
        },
        EXPLAINER_PATH,
    )
    print(f"SHAP explainer saved -> {EXPLAINER_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
