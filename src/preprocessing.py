"""Data loading, cleaning and feature-engineering pipeline.

The preprocessor built here is a scikit-learn ColumnTransformer that is used
*identically* during training and during live prediction so there is no data
leakage and no training/serving skew.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    HEART_CSV,
    NUMERIC_FEATURES,
    TARGET,
)


def load_data(path: str | pd.DataFrame = HEART_CSV) -> pd.DataFrame:
    """Load the dataset from CSV (or accept a DataFrame directly)."""
    if isinstance(path, pd.DataFrame):
        return path.copy()
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows; drop rows with missing target or constant nulls.

    The downloaded dataset is already free of missing values, but we keep the
    cleaning logic defensive so it works with any future dataset refresh.
    """
    cleaned = df.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.dropna(subset=[TARGET])
    cleaned = cleaned[cleaned[TARGET].isin([0, 1])]
    return cleaned


def build_preprocessor() -> Pipeline:
    """Return the full preprocessing pipeline used by every model.

    Numeric features are standardised, categorical features are one-hot encoded
    (with ``handle_unknown='ignore'`` so future unseen categories do not crash
    prediction), and binary 0/1 features pass through unchanged.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    )
    binary_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
        ]
    )
    return Pipeline(steps=[("prep", preprocessor)])


def prepare_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into features (X) and target (y).

    The column order is pinned explicitly so prediction requests are always
    assembled in exactly the same order the models were trained on.
    """
    feature_order = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    )
    X = df[feature_order].copy()
    y = df[TARGET].copy()
    return X, y
