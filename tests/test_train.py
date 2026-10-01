"""End-to-end smoke test: train on synthetic data, then score a customer."""

import json
from pathlib import Path

import pandas as pd

from churn import config
from churn.data import clean
from churn.evaluate import save_figures
from churn.predict import load_model, predict_customer, top_drivers
from churn.train import run


def test_training_end_to_end(raw_csv: Path, tmp_path: Path, raw_df: pd.DataFrame) -> None:
    models_dir, reports_dir = tmp_path / "models", tmp_path / "reports"
    metrics = run(raw_csv, models_dir, reports_dir)

    saved = json.loads((reports_dir / config.METRICS_FILENAME).read_text())
    assert saved["best_model"] in {"logistic_regression", "random_forest", "xgboost"}
    assert 0 <= saved["test"]["f1"] <= 1
    assert saved == json.loads(json.dumps(metrics))

    bundle = load_model(models_dir / config.MODEL_FILENAME)
    profile = clean(raw_df)[config.RAW_FEATURES].iloc[0].to_dict()
    proba, _ = predict_customer(bundle, profile)
    assert 0 <= proba <= 1
    assert len(top_drivers(bundle, n=5)) == 5

    X = clean(raw_df)[config.RAW_FEATURES]
    y = clean(raw_df)[config.TARGET].to_numpy()
    figures = save_figures(y, bundle["pipeline"].predict_proba(X)[:, 1], bundle, reports_dir)
    assert all(path.exists() and path.stat().st_size > 0 for path in figures)
