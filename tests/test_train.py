from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_pipeline.labels import AAMI_CLASSES
from training.train import (
    build_model,
    compute_class_weights,
    encode_labels,
    prepare_training_data,
    split_train_validation,
    train,
)

FEATURE_COLUMNS = [
    "RPeakCount", "SpectralEnergy", "TotalPSD", "WaveletEnergy",
    "ShannonEntropy", "SignalSTD", "Skewness", "Kurtosis", "Variance",
]


def _synthetic_dataset(rows_per_record: int, record_ids: list[int], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for record_id in record_ids:
        for _ in range(rows_per_record):
            row = {col: rng.standard_normal() for col in FEATURE_COLUMNS}
            row["AAMIClass"] = rng.choice(AAMI_CLASSES)
            row["RecordID"] = record_id
            rows.append(row)
    return pd.DataFrame(rows)


class TestSplitTrainValidation:
    def test_no_record_id_appears_in_both_partitions(self):
        rng = np.random.default_rng(0)
        df = _synthetic_dataset(rows_per_record=20, record_ids=list(range(1, 21)), rng=rng)

        train_df, val_df = split_train_validation(df, validation_fraction=0.15, seed=42)

        assert set(train_df["RecordID"]).isdisjoint(set(val_df["RecordID"]))
        assert len(train_df) + len(val_df) == len(df)

    def test_deterministic_with_the_same_seed(self):
        rng = np.random.default_rng(1)
        df = _synthetic_dataset(rows_per_record=10, record_ids=list(range(1, 21)), rng=rng)

        train_a, val_a = split_train_validation(df, validation_fraction=0.15, seed=7)
        train_b, val_b = split_train_validation(df, validation_fraction=0.15, seed=7)

        assert set(train_a["RecordID"]) == set(train_b["RecordID"])
        assert set(val_a["RecordID"]) == set(val_b["RecordID"])

    def test_roughly_respects_the_requested_validation_fraction(self):
        rng = np.random.default_rng(2)
        df = _synthetic_dataset(rows_per_record=50, record_ids=list(range(1, 21)), rng=rng)

        _, val_df = split_train_validation(df, validation_fraction=0.15, seed=42)

        # group-level granularity means this can't be exact, but should be in the ballpark
        assert 0.05 < len(val_df) / len(df) < 0.30


class TestEncodeLabels:
    def test_one_hot_shape_matches_number_of_aami_classes(self):
        encoded = encode_labels(pd.Series(["N", "V", "S"]))
        assert encoded.shape == (3, len(AAMI_CLASSES))

    def test_column_order_matches_aami_classes_order(self):
        encoded = encode_labels(pd.Series(["N"]))
        expected = [1.0 if cls == "N" else 0.0 for cls in AAMI_CLASSES]
        np.testing.assert_array_equal(encoded[0], expected)

    def test_raises_clearly_for_an_unknown_class(self):
        with pytest.raises(ValueError, match="Unknown AAMI class"):
            encode_labels(pd.Series(["N", "X"]))


class TestComputeClassWeights:
    def test_rarer_class_gets_a_higher_weight(self):
        # 100 examples of class 0, 5 of class 1 -- class 1 must be weighted higher
        y = np.array([0] * 100 + [1] * 5)
        weights = compute_class_weights(y)
        assert weights[1] > weights[0]

    def test_balanced_classes_get_equal_weight(self):
        y = np.array([0] * 50 + [1] * 50)
        weights = compute_class_weights(y)
        assert weights[0] == pytest.approx(weights[1])

    def test_only_present_classes_get_a_weight(self):
        # class 2 (e.g. "V") never appears in this partition
        y = np.array([0] * 50 + [1] * 50)
        weights = compute_class_weights(y)
        assert set(weights.keys()) == {0, 1}

    def test_handles_a_class_with_only_2_examples_without_crashing(self):
        # the real, observed case: Q had only 2 examples in one training partition
        y = np.array([0] * 100 + [1] * 20 + [2] * 2)
        weights = compute_class_weights(y)
        assert weights[2] > weights[0]
        assert np.isfinite(weights[2])

    def test_extreme_imbalance_is_capped_not_left_unbounded(self):
        # matches the real, observed ratio (~37782 N : 2 Q): uncapped
        # "balanced" weight would be ~4185, empirically confirmed to blow up
        # training (val_loss ~16, ~10x higher than random-guessing baseline).
        y = np.array([0] * 37782 + [1] * 2)
        weights = compute_class_weights(y, max_weight=25.0)
        assert weights[1] == pytest.approx(25.0)

    def test_weight_below_the_cap_is_left_unchanged(self):
        y = np.array([0] * 100 + [1] * 20)  # raw weight = 120/(2*20) = 3.0, well under the cap
        weights = compute_class_weights(y, max_weight=25.0)
        assert weights[1] == pytest.approx(3.0)


class TestPrepareTrainingData:
    def test_scaler_is_fit_on_the_given_data_only(self):
        rng = np.random.default_rng(3)
        df = _synthetic_dataset(rows_per_record=30, record_ids=[1, 2, 3], rng=rng)

        _, _, scaler, _ = prepare_training_data(df, FEATURE_COLUMNS, seed=42)

        # a fitted StandardScaler exposes mean_/scale_ sized to the feature count
        assert scaler.mean_.shape == (len(FEATURE_COLUMNS),)

    def test_does_not_invent_any_rows(self):
        # No SMOTE: an earlier version balanced classes by oversampling, which
        # amplified a 2-example class to ~38,000 rows on a real run -- purely
        # interpolated between 2 points, not real data. class_weight must not
        # change the row count at all.
        rng = np.random.default_rng(4)
        rows = []
        for _ in range(50):
            rows.append({**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": "N", "RecordID": 1})
        for _ in range(5):
            rows.append({**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": "V", "RecordID": 1})
        df = pd.DataFrame(rows)

        X, y, _, _ = prepare_training_data(df, FEATURE_COLUMNS, seed=42)

        assert X.shape[0] == len(df)
        assert y.shape[0] == len(df)

    def test_handles_a_very_rare_class_without_crashing(self):
        rng = np.random.default_rng(5)
        rows = []
        for _ in range(50):
            rows.append({**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": "N", "RecordID": 1})
        for _ in range(2):  # the real, observed minimum for the Q class
            rows.append({**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": "Q", "RecordID": 1})
        df = pd.DataFrame(rows)

        X, y, _, class_weights = prepare_training_data(df, FEATURE_COLUMNS, seed=42)
        assert X.shape[0] == y.shape[0] == len(df)
        assert all(np.isfinite(w) for w in class_weights.values())


class TestBuildModel:
    def test_input_and_output_shapes(self):
        model = build_model(input_dim=9, num_classes=5)
        assert model.input_shape == (None, 9)
        assert model.output_shape == (None, 5)

    def test_is_compiled_and_ready_to_fit(self):
        model = build_model(input_dim=9, num_classes=5)
        assert model.optimizer is not None
        assert model.loss == "categorical_crossentropy"


class TestTrainSmoke:
    def test_full_pipeline_runs_and_saves_artifacts(self, tmp_path: Path):
        rng = np.random.default_rng(6)
        df = _synthetic_dataset(rows_per_record=30, record_ids=list(range(1, 11)), rng=rng)
        csv_path = tmp_path / "dataset_ds1.csv"
        df.to_csv(csv_path, index=False)

        version_dir = train(csv_path, tmp_path / "models", epochs=1, batch_size=16, seed=42)

        assert (version_dir / "model.keras").exists()
        assert (version_dir / "scaler.pkl").exists()
        assert (version_dir / "training_config.json").exists()
