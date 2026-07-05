from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_pipeline.labels import AAMI_CLASSES
from training.train import (
    _smote_k_neighbors,
    build_model,
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


class TestSmoteKNeighbors:
    def test_uses_default_5_when_classes_are_plentiful(self):
        y = np.array([0] * 100 + [1] * 100)
        assert _smote_k_neighbors(y) == 5

    def test_shrinks_below_the_smallest_class_count(self):
        # only 3 examples of class 1 -- k_neighbors must be < 3
        y = np.array([0] * 100 + [1] * 3)
        k = _smote_k_neighbors(y)
        assert k < 3
        assert k >= 1

    def test_never_returns_less_than_1(self):
        y = np.array([0] * 100 + [1] * 2)
        assert _smote_k_neighbors(y) >= 1


class TestPrepareTrainingData:
    def test_scaler_is_fit_on_the_given_data_only(self):
        rng = np.random.default_rng(3)
        df = _synthetic_dataset(rows_per_record=30, record_ids=[1, 2, 3], rng=rng)

        _, _, scaler = prepare_training_data(df, FEATURE_COLUMNS, seed=42)

        # a fitted StandardScaler exposes mean_/scale_ sized to the feature count
        assert scaler.mean_.shape == (len(FEATURE_COLUMNS),)

    def test_output_classes_are_balanced_after_smote(self):
        rng = np.random.default_rng(4)
        rows = []
        # deliberately imbalanced: 50 "N", 5 "V"
        for _ in range(50):
            rows.append({**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": "N", "RecordID": 1})
        for _ in range(5):
            rows.append({**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": "V", "RecordID": 1})
        df = pd.DataFrame(rows)

        X, y, _ = prepare_training_data(df, FEATURE_COLUMNS, seed=42)

        class_counts = y.sum(axis=0)
        present_classes = class_counts[class_counts > 0]
        assert len(set(present_classes)) == 1  # every present class has equal count after SMOTE
        assert X.shape[0] == y.shape[0]

    def test_handles_a_very_rare_class_without_crashing(self):
        rng = np.random.default_rng(5)
        rows = []
        for _ in range(50):
            rows.append({**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": "N", "RecordID": 1})
        for _ in range(2):  # fewer than SMOTE's default k_neighbors=5 requires
            rows.append({**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": "Q", "RecordID": 1})
        df = pd.DataFrame(rows)

        X, y, _ = prepare_training_data(df, FEATURE_COLUMNS, seed=42)
        assert X.shape[0] == y.shape[0]


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
