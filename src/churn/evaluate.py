"""Metrics, threshold tuning and evaluation of the saved model on the test set."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn import config, data, predict

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Classification metrics for the churn class at a given threshold."""
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Return the probability cut-off that maximises F1 on the given data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # The last precision/recall pair has no matching threshold.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def main() -> None:
    """Re-evaluate the saved model on the held-out test split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.RAW_DATA_PATH)
    parser.add_argument("--model", type=Path, default=config.MODELS_DIR / config.MODEL_FILENAME)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    df = data.clean(data.load_raw(args.data))
    X, y = data.split_xy(df)
    _, X_test, _, y_test = data.train_test(X, y)

    bundle = predict.load_model(args.model)
    proba = predict.predict_proba(bundle, X_test)
    y_pred = (proba >= bundle["threshold"]).astype(int)

    metrics = compute_metrics(y_test.to_numpy(), proba, bundle["threshold"])
    logger.info("Model: %s, threshold %.3f", bundle["model_name"], bundle["threshold"])
    logger.info("Test metrics: %s", json.dumps(metrics))
    logger.info("Confusion matrix [[TN, FP], [FN, TP]]:\n%s", confusion_matrix(y_test, y_pred))
    logger.info(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, target_names=["No churn", "Churn"]),
    )


if __name__ == "__main__":
    main()
