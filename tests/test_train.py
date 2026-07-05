from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_class_weight

from ecg_pipeline.labels import AAMI_CLASSES
from training.train import (
    build_model,
    compute_class_weights,
    prepare_training_data,
    split_train_validation,
    train,
    warn_on_missing_classes,
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


class TestWarnOnMissingClasses:
    def test_logs_a_warning_when_a_class_is_absent_from_training(self, caplog):
        full_df = pd.DataFrame({"AAMIClass": ["N", "N", "Q", "Q"], "RecordID": [1, 1, 2, 2]})
        train_df = full_df[full_df["RecordID"] == 1]  # drops record 2 -- Q disappears entirely

        with caplog.at_level("WARNING"):
            warn_on_missing_classes(full_df, train_df)

        assert "Q" in caplog.text
        assert "ABSENT" in caplog.text

    def test_no_warning_when_every_class_survives_the_split(self, caplog):
        full_df = pd.DataFrame({"AAMIClass": ["N", "Q"], "RecordID": [1, 2]})
        train_df = full_df  # nothing dropped

        with caplog.at_level("WARNING"):
            warn_on_missing_classes(full_df, train_df)

        assert caplog.text == ""


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

    def test_extreme_outlier_is_capped_at_the_second_highest_natural_weight(self):
        # matches the real, observed ratio (~37782 N : 2808 V : 2 Q): Q's
        # uncapped "balanced" weight would be ~2500x V's -- empirically
        # confirmed at this kind of ratio to blow up training (val_loss ~16,
        # ~10x the random-guessing baseline). The cap must come from V's
        # own naturally-occurring weight, not a memorized constant.
        y = np.array([0] * 37782 + [1] * 2808 + [2] * 2)
        raw_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y)
        second_highest_raw_weight = np.sort(raw_weights)[-2]

        weights = compute_class_weights(y)

        assert weights[2] == pytest.approx(second_highest_raw_weight)
        assert weights[2] < 100  # nowhere near the uncapped ~6800 it would otherwise be

    def test_weights_below_the_cap_are_left_unchanged(self):
        # only 2 classes: nothing to cap against (no "second-highest" to
        # derive from), so both weights must pass through as sklearn computed them
        y = np.array([0] * 100 + [1] * 20)
        weights = compute_class_weights(y)
        assert weights[1] == pytest.approx(120 / (2 * 20))
        assert weights[0] == pytest.approx(120 / (2 * 100))

    def test_cap_self_adjusts_to_a_different_data_distribution(self):
        # same extreme rare-class ratio, but a much rarer "second-highest"
        # class than the previous test -- the cap must track it, not stay
        # fixed at whatever a previous run's data happened to produce.
        y = np.array([0] * 100_000 + [1] * 50 + [2] * 2)
        weights = compute_class_weights(y)
        raw_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y)
        assert weights[2] == pytest.approx(np.sort(raw_weights)[-2])


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

    def test_raises_on_an_unrecognized_class_instead_of_silently_becoming_n(self):
        # Regression test for a real bug: prepare_training_data used to call
        # a bare .map() with no validation, unlike encode_labels next to it.
        # An unrecognized class mapped to NaN, which promoted y_indices to
        # float64; keras.utils.to_categorical then cast that NaN to int64,
        # which NumPy silently resolves to 0 -- i.e. a corrupted label got
        # silently trained as class "N" with no error and no log line.
        rng = np.random.default_rng(7)
        rows = [
            {**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": "N", "RecordID": 1}
            for _ in range(10)
        ]
        rows.append({**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": "TYPO", "RecordID": 1})
        df = pd.DataFrame(rows)

        with pytest.raises(ValueError, match="Unknown AAMI class"):
            prepare_training_data(df, FEATURE_COLUMNS, seed=42)


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
