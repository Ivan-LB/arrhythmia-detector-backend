from pathlib import Path

import numpy as np
import pytest

from training.evaluate import (
    build_metrics_table,
    calculate_class_metrics,
    evaluate_predictions,
)

# Hand-verified 3-class confusion matrix (rows=true, cols=predicted):
#        pred0 pred1 pred2
# true0    50    3     2      (55 total)
# true1     4   30     1      (35 total)
# true2     2    1    20      (23 total)
# total = 113
CM = np.array([
    [50, 3, 2],
    [4, 30, 1],
    [2, 1, 20],
])


class TestCalculateClassMetrics:
    def test_class_0_metrics_match_hand_computed_values(self):
        metrics = calculate_class_metrics(CM, class_index=0)
        # TP=50, FP=6, FN=5, TN=52
        assert metrics.sensitivity == pytest.approx(50 / 55)
        assert metrics.specificity == pytest.approx(52 / 58)
        assert metrics.positive_predictivity == pytest.approx(50 / 56)
        assert metrics.negative_predictivity == pytest.approx(52 / 57)

    def test_class_1_metrics_match_hand_computed_values(self):
        metrics = calculate_class_metrics(CM, class_index=1)
        # TP=30, FP=4, FN=5, TN=74
        assert metrics.sensitivity == pytest.approx(30 / 35)
        assert metrics.specificity == pytest.approx(74 / 78)
        assert metrics.positive_predictivity == pytest.approx(30 / 34)
        assert metrics.negative_predictivity == pytest.approx(74 / 79)

    def test_returns_zero_not_nan_for_a_class_with_no_true_or_predicted_examples(self):
        # a class that never appears as true or predicted -> 0/0 guards must hold
        empty_cm = np.array([[10, 0], [0, 0]])
        metrics = calculate_class_metrics(empty_cm, class_index=1)
        assert metrics.sensitivity == 0
        assert metrics.positive_predictivity == 0
        assert np.isfinite(metrics.specificity)
        assert np.isfinite(metrics.negative_predictivity)


class TestBuildMetricsTable:
    def test_has_one_row_per_class_with_correct_columns(self):
        table = build_metrics_table(CM, class_names=["N", "S", "V"])
        assert list(table.index) == ["N", "S", "V"]
        assert set(table.columns) == {"Sensitivity", "Specificity", "PositivePredictivity", "NegativePredictivity"}

    def test_values_match_calculate_class_metrics(self):
        table = build_metrics_table(CM, class_names=["N", "S", "V"])
        direct = calculate_class_metrics(CM, class_index=0)
        assert table.loc["N", "Sensitivity"] == pytest.approx(direct.sensitivity)


class TestEvaluatePredictions:
    def test_includes_confusion_matrix_accuracy_and_per_class_table(self):
        y_true = np.array([0] * 55 + [1] * 35 + [2] * 23)
        # construct predictions that reproduce the exact CM above
        y_pred = (
            [0] * 50 + [1] * 3 + [2] * 2
            + [0] * 4 + [1] * 30 + [2] * 1
            + [0] * 2 + [1] * 1 + [2] * 20
        )
        y_pred = np.array(y_pred)

        result = evaluate_predictions(y_true, y_pred, class_names=["N", "S", "V"])

        np.testing.assert_array_equal(result["confusion_matrix"], CM)
        assert result["accuracy"] == pytest.approx((50 + 30 + 20) / 113)
        assert result["per_class_metrics"].loc["N", "Sensitivity"] == pytest.approx(50 / 55)

    def test_accuracy_is_1_for_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        result = evaluate_predictions(y_true, y_pred, class_names=["N", "S", "V"])
        assert result["accuracy"] == pytest.approx(1.0)


class TestEvaluateIntegration:
    def test_evaluates_a_freshly_trained_model_against_synthetic_ds2_data(self, tmp_path: Path):
        import pandas as pd

        from training.train import train

        feature_columns = [
            "RPeakCount", "SpectralEnergy", "TotalPSD", "WaveletEnergy",
            "ShannonEntropy", "SignalSTD", "Skewness", "Kurtosis", "Variance",
        ]
        rng = np.random.default_rng(0)

        def _synthetic_df(record_ids):
            rows = []
            for record_id in record_ids:
                for _ in range(30):
                    row = {c: rng.standard_normal() for c in feature_columns}
                    row["AAMIClass"] = rng.choice(["N", "S", "V", "F", "Q"])
                    row["RecordID"] = record_id
                    rows.append(row)
            return pd.DataFrame(rows)

        ds1_csv = tmp_path / "dataset_ds1.csv"
        ds2_csv = tmp_path / "dataset_ds2.csv"
        _synthetic_df(range(1, 11)).to_csv(ds1_csv, index=False)
        _synthetic_df(range(100, 105)).to_csv(ds2_csv, index=False)

        from training.evaluate import evaluate

        model_dir = train(ds1_csv, tmp_path / "models", epochs=1, batch_size=16, seed=42)
        metrics = evaluate(model_dir, ds2_csv)

        assert (model_dir / "metrics.json").exists()
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["confusion_matrix"].shape == (5, 5)
