from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.inference import diagnose_beats, load_inference_bundle
from ecg_pipeline.labels import AAMI_CLASSES

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_COLUMNS = [
    "RPeakCount", "SpectralEnergy", "TotalPSD", "WaveletEnergy",
    "ShannonEntropy", "SignalSTD", "Skewness", "Kurtosis", "Variance",
]


def _train_a_tiny_model(tmp_path: Path):
    from training.train import train

    rng = np.random.default_rng(0)
    rows = [
        {**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": rng.choice(AAMI_CLASSES), "RecordID": rid}
        for rid in range(1, 11)
        for _ in range(30)
    ]
    ds1_csv = tmp_path / "dataset_ds1.csv"
    pd.DataFrame(rows).to_csv(ds1_csv, index=False)
    return train(ds1_csv, tmp_path / "models", epochs=1, batch_size=16, seed=42)


class TestLoadInferenceBundle:
    def test_loads_a_real_trained_model_and_scaler(self, tmp_path: Path):
        model_dir = _train_a_tiny_model(tmp_path)
        bundle = load_inference_bundle(model_dir)

        assert bundle.model_version == model_dir.name
        assert bundle.model is not None
        assert bundle.scaler is not None

    def test_raises_clearly_when_model_file_is_missing(self, tmp_path: Path):
        empty_dir = tmp_path / "no-model-here"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="model.keras"):
            load_inference_bundle(empty_dir)

    def test_raises_clearly_when_scaler_file_is_missing(self, tmp_path: Path):
        model_dir = _train_a_tiny_model(tmp_path)
        (model_dir / "scaler.pkl").unlink()
        with pytest.raises(FileNotFoundError, match="scaler.pkl"):
            load_inference_bundle(model_dir)


class TestDiagnoseBeats:
    FS = 360.0

    def _signal_with_sine_beats_at(self, centers):
        length = max(centers) + 1000
        signal = np.full(length, 0.5)
        for center in centers:
            t = np.arange(center - 180, center + 180) / self.FS
            signal[center - 180 : center + 180] += 0.2 * np.sin(2 * np.pi * 10.0 * t)
        return signal

    def test_returns_one_result_per_valid_beat_with_a_confident_valid_class(self, tmp_path: Path):
        model_dir = _train_a_tiny_model(tmp_path)
        bundle = load_inference_bundle(model_dir)
        signal = self._signal_with_sine_beats_at([1000, 2000, 3000])

        results = diagnose_beats(bundle, signal, self.FS, np.array([1000, 2000, 3000]))

        assert len(results) == 3
        for result in results:
            assert result.aami_class in AAMI_CLASSES
            assert 0.0 <= result.confidence <= 1.0

    def test_skips_beats_whose_window_runs_off_the_signal_edge(self, tmp_path: Path):
        model_dir = _train_a_tiny_model(tmp_path)
        bundle = load_inference_bundle(model_dir)
        signal = self._signal_with_sine_beats_at([1000])

        results = diagnose_beats(bundle, signal, self.FS, np.array([1000, 5]))  # 5 is too close to the start

        assert len(results) == 1
        assert results[0].sample_index == 1000

    def test_returns_empty_list_for_no_valid_beats_rather_than_erroring(self, tmp_path: Path):
        model_dir = _train_a_tiny_model(tmp_path)
        bundle = load_inference_bundle(model_dir)
        signal = self._signal_with_sine_beats_at([1000])

        results = diagnose_beats(bundle, signal, self.FS, np.array([5]))  # only an out-of-bounds sample

        assert results == []
