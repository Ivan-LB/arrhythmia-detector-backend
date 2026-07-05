"""Evaluate a trained beat classifier on DS2 -- the held-out set touched
nowhere else in this pipeline.

Reports per-class Sensitivity, Specificity, Positive Predictivity, and
Negative Predictivity, not just overall accuracy: accuracy alone is
misleading here since Normal beats dominate MIT-BIH so heavily that a
model could score high accuracy while missing most S/V/F/Q beats
entirely, which are the clinically important ones. Standard practice
throughout the inter-patient MIT-BIH literature.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from ecg_pipeline import features as features_module
from ecg_pipeline.labels import AAMI_CLASSES
from training.class_encoding import class_series_to_indices

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: tuple[str, ...] = features_module.FEATURE_NAMES


@dataclass(frozen=True)
class ClassMetrics:
    sensitivity: float
    specificity: float
    positive_predictivity: float
    negative_predictivity: float


def calculate_class_metrics(cm: npt.NDArray[np.int64], class_index: int) -> ClassMetrics:
    """Sensitivity/Specificity/Positive Predictivity/Negative Predictivity
    for one class from a confusion matrix (rows=true, cols=predicted).

    Ported from the original project's calculate_metrics function -- that
    part of the original code was already correct and worth keeping, just
    given type hints and a proper return type instead of a bare tuple.
    """
    true_positive = cm[class_index, class_index]
    false_positive = cm[:, class_index].sum() - true_positive
    false_negative = cm[class_index, :].sum() - true_positive
    true_negative = cm.sum() - (true_positive + false_positive + false_negative)

    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) != 0 else 0
    ppv = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    npv = true_negative / (true_negative + false_negative) if (true_negative + false_negative) != 0 else 0

    return ClassMetrics(
        sensitivity=sensitivity,
        specificity=specificity,
        positive_predictivity=ppv,
        negative_predictivity=npv,
    )


def build_metrics_table(cm: npt.NDArray[np.int64], class_names: list[str]) -> pd.DataFrame:
    """Per-class Se/Sp/PPV/NPV table, one row per class name in order."""
    rows = [asdict(calculate_class_metrics(cm, index)) for index in range(len(class_names))]
    table = pd.DataFrame(rows, index=class_names)
    table.columns = ["Sensitivity", "Specificity", "PositivePredictivity", "NegativePredictivity"]
    return table


def evaluate_predictions(
    y_true_indices: npt.NDArray[np.int64],
    y_pred_indices: npt.NDArray[np.int64],
    class_names: list[str],
) -> dict:
    """Confusion matrix, overall accuracy, and the per-class metrics table."""
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=list(range(len(class_names))))
    accuracy = float(np.trace(cm) / cm.sum())
    return {
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "per_class_metrics": build_metrics_table(cm, class_names),
    }


def evaluate(model_dir: Path, ds2_csv_path: Path) -> dict:
    """Load a trained model + scaler, evaluate on DS2, save metrics.json.

    The scaler is only ever .transform()'d here, never re-fit -- DS2 must
    never influence the scaler's statistics.
    """
    model = keras.models.load_model(model_dir / "model.keras")
    scaler = joblib.load(model_dir / "scaler.pkl")

    df = pd.read_csv(ds2_csv_path)
    if len(df) == 0:
        raise ValueError(
            f"{ds2_csv_path} has 0 rows -- nothing to evaluate. "
            "This can happen with an empty-but-valid dataset CSV (e.g. a "
            "DS1-only record subset); check the file before evaluating it."
        )

    feature_columns = list(FEATURE_COLUMNS)
    X = scaler.transform(df[feature_columns].to_numpy())
    y_true_indices = class_series_to_indices(df["AAMIClass"])

    y_pred_probabilities = model.predict(X, verbose=0)
    y_pred_indices = np.argmax(y_pred_probabilities, axis=1)

    result = evaluate_predictions(y_true_indices, y_pred_indices, class_names=list(AAMI_CLASSES))

    metrics_json = {
        "accuracy": result["accuracy"],
        "confusion_matrix": result["confusion_matrix"].tolist(),
        "class_names": list(AAMI_CLASSES),
        "per_class_metrics": result["per_class_metrics"].to_dict(orient="index"),
        "ds2_rows_evaluated": len(df),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    logger.info("Accuracy: %.4f", result["accuracy"])
    logger.info("\n%s", result["per_class_metrics"].to_string())

    return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_dir = Path(sys.argv[1])
    ds2_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("Data/dataset_ds2.csv")
    evaluate(model_dir, ds2_csv)
