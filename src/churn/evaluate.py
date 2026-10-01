"""Metrics, threshold tuning and evaluation of the saved model on the test set."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
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


def save_figures(
    y_true: np.ndarray, proba: np.ndarray, bundle: predict.ModelBundle, out_dir: Path
) -> list[Path]:
    """Write confusion matrix, ROC curve and feature importance PNGs for the README."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = bundle["threshold"]
    y_pred = (proba >= threshold).astype(int)
    paths = []

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No churn", "Churn"], cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(f"Test confusion matrix (threshold {threshold:.2f})")
    paths.append(out_dir / "confusion_matrix.png")

    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, proba, name=bundle["model_name"], ax=ax2)
    ax2.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Chance")
    ax2.set_title("Test ROC curve")
    ax2.legend(loc="lower right")
    paths.append(out_dir / "roc_curve.png")

    drivers = predict.top_drivers(bundle, n=12).iloc[::-1]
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    ax3.barh(drivers.index, drivers.values, color="#3b6ea5")
    ax3.set_xlabel("importance")
    ax3.set_title(f"Top features ({bundle['model_name']})")
    paths.append(out_dir / "feature_importance.png")

    for figure, path in zip([fig, fig2, fig3], paths):
        figure.tight_layout()
        figure.savefig(path, dpi=100)
        plt.close(figure)
    return paths


def main() -> None:
    """Re-evaluate the saved model on the held-out test split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.RAW_DATA_PATH)
    parser.add_argument("--model", type=Path, default=config.MODELS_DIR / config.MODEL_FILENAME)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
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
    for path in save_figures(y_test.to_numpy(), proba, bundle, args.reports_dir):
        logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
