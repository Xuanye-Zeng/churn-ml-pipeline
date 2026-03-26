import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true: pd.Series, predictions, probabilities) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
    }


def probabilities_to_predictions(probabilities, threshold: float = 0.5):
    return (np.asarray(probabilities) >= threshold).astype(int)


def summarize_cv_scores(cv_scores: dict) -> dict:
    summary = {}
    for metric_name, values in cv_scores.items():
        values_array = np.asarray(values, dtype=float)
        summary[metric_name] = {
            "mean": round(float(values_array.mean()), 4),
            "std": round(float(values_array.std()), 4),
        }
    return summary


def build_threshold_report(y_true: pd.Series, probabilities, thresholds=None) -> dict:
    threshold_values = thresholds or [round(value, 2) for value in np.arange(0.1, 0.91, 0.05)]
    rows = []

    for threshold in threshold_values:
        predictions = probabilities_to_predictions(probabilities, threshold=threshold)
        metrics = compute_metrics(y_true, predictions, probabilities)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
            }
        )

    best_entry = max(rows, key=lambda row: row["f1"])
    default_entry = min(rows, key=lambda row: abs(row["threshold"] - 0.5))

    return {
        "best_f1_threshold": best_entry["threshold"],
        "best_f1_metrics": best_entry,
        "default_threshold_metrics": default_entry,
        "thresholds": rows,
    }
