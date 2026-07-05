"""Model/scaler loading and the diagnose-a-record pipeline.

Reuses ecg_pipeline (the same preprocessing/feature-extraction code the
training pipeline uses) so there is exactly one implementation of that
logic shared between offline training and this live inference path -- see
docs/system-design.md sec 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
from tensorflow import keras

from ecg_pipeline import features as features_module
from ecg_pipeline import preprocessing
from ecg_pipeline.labels import AAMI_CLASSES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceBundle:
    model: keras.Model
    scaler: object
    model_version: str


def load_inference_bundle(model_dir: Path) -> InferenceBundle:
    """Load the model + scaler from a specific, named version directory.

    Raises FileNotFoundError with a clear message if either file is
    missing -- callers (the API's startup lifespan) must let this fail
    loudly rather than serve requests with a missing/partial model. This
    replaces the original project's behavior of loading at bare module
    import time with no error handling at all.
    """
    model_path = model_dir / "model.keras"
    scaler_path = model_dir / "scaler.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No model.keras found at {model_dir}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"No scaler.pkl found at {model_dir}")

    logger.info("Loading model from %s", model_dir)
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return InferenceBundle(model=model, scaler=scaler, model_version=model_dir.name)


@dataclass(frozen=True)
class BeatResult:
    sample_index: int
    aami_class: str
    confidence: float


def diagnose_beats(
    bundle: InferenceBundle,
    signal: npt.NDArray[np.float64],
    fs: float,
    beat_samples: npt.NDArray[np.int64],
) -> list[BeatResult]:
    """Run feature extraction + inference for a list of beat center samples
    (either real annotations or ecg_pipeline.preprocessing.detect_r_peaks
    output).

    Beats whose window runs off the recording's edge, or whose window is
    degenerate (no real signal variation), are skipped -- same policy as
    the training dataset builder: not fabricating a result for a beat with
    no real signal to classify. Predictions are batched into a single
    model.predict() call rather than one per beat, since a half-hour
    recording can have thousands of beats.
    """
    filtered = preprocessing.apply_notch_filter(preprocessing.remove_baseline(signal), fs)

    valid_samples: list[int] = []
    feature_rows: list[npt.NDArray[np.float64]] = []
    for sample in beat_samples:
        window = preprocessing.window_around_sample(filtered, int(sample), fs)
        if window is None:
            continue
        normalized = preprocessing.min_max_normalize(window)
        try:
            beat_features = features_module.extract_features(normalized, fs)
        except ValueError:
            continue
        valid_samples.append(int(sample))
        feature_rows.append(beat_features.to_array())

    if not feature_rows:
        return []

    scaled = bundle.scaler.transform(np.array(feature_rows))
    probabilities = bundle.model.predict(scaled, verbose=0)
    class_indices = np.argmax(probabilities, axis=1)

    return [
        BeatResult(
            sample_index=sample,
            aami_class=AAMI_CLASSES[class_index],
            confidence=float(probabilities[row_index, class_index]),
        )
        for row_index, (sample, class_index) in enumerate(zip(valid_samples, class_indices))
    ]
