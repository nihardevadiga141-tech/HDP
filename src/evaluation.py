"""Model evaluation helpers.

All metrics are computed from *actual* test-set predictions — nothing is
fabricated. ROC-curve point arrays are exported as JSON so the Streamlit
performance page can plot the real curves without holding the dataset.
"""
from __future__ import annotations

import json
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import METRICS_JSON, PLOTS_DIR


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Evaluate a fitted classifier (pipeline) on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics


def roc_points(model, X_test, y_test) -> dict:
    """Return fpr/tpr/threshold arrays for the ROC curve."""
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    return {
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
        "thresholds": [float(x) for x in thresholds],
    }


def save_roc_curves(roc_data: dict[str, dict], path=PLOTS_DIR / "roc_curves.png") -> None:
    """Save a combined ROC plot for all models (one figure per model)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    for name, points in roc_data.items():
        ax.plot(
            points["fpr"],
            points["tpr"],
            label=f"{name} (AUC={points['roc_auc']:.3f})",
            linewidth=2,
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on the Test Set")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_matrices(cm_data: dict[str, dict], path=PLOTS_DIR / "confusion_matrices.png") -> None:
    """Save a grid of confusion-matrix heatmaps for all models."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cm_data)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.5 * rows))
    axes = [axes] if n == 1 else list(axes.flatten())

    for ax, (name, cm) in zip(axes, cm_data.items()):
        matrix = [
            [cm["tn"], cm["fp"]],
            [cm["fn"], cm["tp"]],
        ]
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No disease", "Disease"])
        ax.set_yticklabels(["No disease", "Disease"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    matrix[i][j],
                    ha="center",
                    va="center",
                    color="white" if matrix[i][j] > 40 else "black",
                    fontsize=16,
                    fontweight="bold",
                )
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes[len(cm_data):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_metrics(metrics_by_model: dict[str, dict], roc_by_model: dict[str, dict]) -> None:
    """Persist metrics + ROC arrays to JSON."""
    payload = {
        "models": {
            name: {**metrics, "roc": roc_by_model[name]}
            for name, metrics in metrics_by_model.items()
        }
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_metrics() -> dict:
    """Load saved metrics JSON (returns {} if unavailable)."""
    if not METRICS_JSON.exists():
        return {}
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))
