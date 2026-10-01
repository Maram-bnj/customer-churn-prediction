"""Benchmark models with cross-validation, tune the threshold and save the best one.

Usage: python -m churn.train [--data PATH] [--models-dir DIR] [--reports-dir DIR]
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import config, data
from churn.evaluate import best_f1_threshold, compute_metrics
from churn.features import build_preprocessor

logger = logging.getLogger(__name__)


def make_pipeline(model: ClassifierMixin) -> Pipeline:
    """Preprocessing + classifier, so the saved model accepts raw customer rows."""
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])


def candidate_models(y_train: pd.Series) -> dict[str, ClassifierMixin]:
    """Models to compare. Class imbalance is handled with class weights."""
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
        ),
    }


def run(data_path: Path, models_dir: Path, reports_dir: Path) -> dict:
    """Train, compare and save; returns the metrics written to metrics.json."""
    df = data.clean(data.load_raw(data_path))
    X, y = data.split_xy(df)
    X_train, X_test, y_train, y_test = data.train_test(X, y)
    logger.info(
        "Train: %d rows, test: %d rows, churn rate %.1f%%",
        len(X_train), len(X_test), 100 * y.mean(),
    )

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    cv_results = {}
    for name, model in candidate_models(y_train).items():
        scores = cross_validate(
            make_pipeline(model), X_train, y_train, cv=cv, scoring=["f1", "roc_auc", "accuracy"]
        )
        cv_results[name] = {
            "f1_mean": round(scores["test_f1"].mean(), 4),
            "f1_std": round(scores["test_f1"].std(), 4),
            "roc_auc_mean": round(scores["test_roc_auc"].mean(), 4),
            "accuracy_mean": round(scores["test_accuracy"].mean(), 4),
        }
        logger.info(
            "%-20s CV F1 %.3f +/- %.3f",
            name, cv_results[name]["f1_mean"], cv_results[name]["f1_std"],
        )

    best_name = max(cv_results, key=lambda name: cv_results[name]["f1_mean"])
    best_pipeline = make_pipeline(candidate_models(y_train)[best_name])
    logger.info("Best model by CV F1: %s", best_name)

    # Tune the decision threshold on out-of-fold predictions, so the test set stays unseen.
    oof_proba = cross_val_predict(
        best_pipeline, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    threshold = best_f1_threshold(y_train.to_numpy(), oof_proba)
    logger.info("Threshold maximising out-of-fold F1: %.3f", threshold)

    best_pipeline.fit(X_train, y_train)
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test.to_numpy(), test_proba, threshold)
    test_metrics_default = compute_metrics(y_test.to_numpy(), test_proba, 0.5)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_accuracy = round(baseline.score(X_test, y_test), 4)
    accuracy_gain_pct = round(
        100 * (test_metrics["accuracy"] - baseline_accuracy) / baseline_accuracy, 2
    )
    accuracy_gain_points = round(100 * (test_metrics["accuracy"] - baseline_accuracy), 2)
    test_pred = (test_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    metrics = {
        "dataset_rows": len(df),
        "cv_folds": config.CV_FOLDS,
        "cross_validation": cv_results,
        "best_model": best_name,
        "threshold": round(threshold, 4),
        "test": test_metrics,
        "test_at_0.5": test_metrics_default,
        "baseline": {"model": "majority class", "accuracy": baseline_accuracy},
        # Relative gain: (model - baseline) / baseline. Points: model - baseline, in %.
        "accuracy_gain_vs_baseline_pct": accuracy_gain_pct,
        "accuracy_gain_vs_baseline_points": accuracy_gain_points,
        "test_rows": len(y_test),
        "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / config.MODEL_FILENAME
    joblib.dump(
        {"pipeline": best_pipeline, "threshold": threshold, "model_name": best_name}, model_path
    )
    metrics_path = reports_dir / config.METRICS_FILENAME
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved model to %s and metrics to %s", model_path, metrics_path)
    return metrics


def print_summary(metrics: dict) -> None:
    """Print the results table used in the README."""
    test = metrics["test"]
    rows = [
        ("Best model", metrics["best_model"]),
        ("Decision threshold", f"{metrics['threshold']:.2f}"),
        ("Test F1 (churn class)", f"{test['f1']:.2f}"),
        ("Test weighted F1", f"{test['f1_weighted']:.2f}"),
        ("Test precision / recall", f"{test['precision']:.2f} / {test['recall']:.2f}"),
        ("Test ROC AUC", f"{test['roc_auc']:.2f}"),
        ("Test accuracy", f"{test['accuracy']:.2f}"),
        ("Baseline accuracy (majority class)", f"{metrics['baseline']['accuracy']:.2f}"),
        (
            "Accuracy gain vs baseline",
            f"{metrics['accuracy_gain_vs_baseline_points']:+.1f} pts "
            f"({metrics['accuracy_gain_vs_baseline_pct']:+.1f}% relative)",
        ),
    ]
    print("\n| Model | CV F1 (mean +/- std) | CV ROC AUC |")
    print("|---|---|---|")
    for name, s in metrics["cross_validation"].items():
        print(f"| {name} | {s['f1_mean']:.3f} +/- {s['f1_std']:.3f} | {s['roc_auc_mean']:.3f} |")
    print("\n| Metric | Value |")
    print("|---|---|")
    for label, value in rows:
        print(f"| {label} | {value} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare churn models.")
    parser.add_argument("--data", type=Path, default=config.RAW_DATA_PATH)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print_summary(run(args.data, args.models_dir, args.reports_dir))


if __name__ == "__main__":
    main()
