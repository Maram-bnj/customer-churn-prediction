"""Load the trained model and score new customers."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from churn import config

ModelBundle = dict[str, Any]


def load_model(path: Path = config.MODELS_DIR / config.MODEL_FILENAME) -> ModelBundle:
    """Load the bundle saved by churn.train (pipeline, threshold, model name)."""
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `python -m churn.train` first.")
    return joblib.load(path)


def predict_proba(bundle: ModelBundle, customers: pd.DataFrame) -> np.ndarray:
    """Churn probability for each row of raw customer data."""
    return bundle["pipeline"].predict_proba(customers[config.RAW_FEATURES])[:, 1]


def predict_customer(bundle: ModelBundle, profile: dict[str, Any]) -> tuple[float, bool]:
    """Score a single customer profile; returns (probability, is_likely_churner)."""
    proba = float(predict_proba(bundle, pd.DataFrame([profile]))[0])
    return proba, proba >= bundle["threshold"]


def top_drivers(bundle: ModelBundle, n: int = 10) -> pd.Series:
    """Global feature importance of the trained model, largest first.

    Uses feature_importances_ for tree models and absolute coefficients for
    logistic regression (features are standardised, so they are comparable).
    """
    pipeline = bundle["pipeline"]
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].named_steps["encode"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model.coef_[0])
    return pd.Series(values, index=names).sort_values(ascending=False).head(n)
