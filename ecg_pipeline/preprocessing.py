"""Signal preprocessing: channel selection, filtering, and beat-centered windowing.

Operates on plain numpy arrays, not wfdb Record objects -- callers (the
dataset builder, the inference API) are responsible for reading the record
and passing in the signal array and lead names. Keeps this module testable
without a wfdb/MIT-BIH dependency.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from scipy.signal import filtfilt, find_peaks, iirnotch

logger = logging.getLogger(__name__)

NOTCH_FREQ_HZ = 60.0
NOTCH_QUALITY = 30.0
WINDOW_SECONDS = 1.0
R_PEAK_MIN_DISTANCE_SECONDS = 0.2
R_PEAK_AMPLITUDE_STD_MULTIPLIER = 2.5


def find_mlii_channel(signal_names: list[str]) -> int:
    """Return the index of the MLII lead among a record's signal names.

    Raises ValueError if MLII isn't present. Callers must handle this by
    skipping the record entirely -- the previous implementation printed a
    warning and fell through to reference an unassigned signal variable,
    raising UnboundLocalError deep inside a thread with no propagation.
    """
    for index, name in enumerate(signal_names):
        if "MLII" in name:
            return index
    raise ValueError(f"MLII channel not found among signal names: {signal_names}")


def remove_baseline(signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Remove DC offset by subtracting the mean."""
    return signal - np.mean(signal)


def apply_notch_filter(signal: npt.NDArray[np.float64], fs: float) -> npt.NDArray[np.float64]:
    """Suppress 60 Hz powerline interference with a zero-phase notch filter.

    Uses filtfilt (forward-backward, zero-phase) rather than a causal
    lfilter: this preprocessing runs entirely offline, and beat windows are
    centered on the annotation's own sample index (see
    window_around_sample), so any phase delay the filter introduces would
    shift signal content relative to that fixed center point. filtfilt
    avoids that at negligible extra cost for a batch/offline pipeline.
    """
    if fs <= 2 * NOTCH_FREQ_HZ:
        raise ValueError(
            f"fs must exceed {2 * NOTCH_FREQ_HZ} Hz (2x the {NOTCH_FREQ_HZ} Hz notch frequency) "
            f"for a valid notch filter; got fs={fs}"
        )
    b, a = iirnotch(NOTCH_FREQ_HZ, NOTCH_QUALITY, fs)
    return filtfilt(b, a, signal)


def window_around_sample(
    signal: npt.NDArray[np.float64],
    center_sample: int,
    fs: float,
    width_seconds: float = WINDOW_SECONDS,
) -> npt.NDArray[np.float64] | None:
    """Extract a fixed-width window centered on a given sample index.

    The center is meant to be an annotation's own sample position, not a
    self-detected peak -- see docs/data-pipeline-architecture.md sec 4 for
    why. Returns None if the fixed-width window would run off either edge
    of the signal, rather than silently returning a shorter window: downstream
    features assume a consistent window length.
    """
    if width_seconds <= 0:
        raise ValueError(f"width_seconds must be positive; got {width_seconds}")
    if fs <= 0:
        raise ValueError(f"fs must be positive; got {fs}")

    half_width = int(width_seconds / 2 * fs)
    start = center_sample - half_width
    end = center_sample + half_width
    if start < 0 or end > len(signal):
        return None
    return signal[start:end]


def detect_r_peaks(
    signal: npt.NDArray[np.float64],
    fs: float,
    min_distance_seconds: float = R_PEAK_MIN_DISTANCE_SECONDS,
    amplitude_std_multiplier: float = R_PEAK_AMPLITUDE_STD_MULTIPLIER,
) -> npt.NDArray[np.intp]:
    """Detect R-peaks in a filtered signal with no ground-truth annotations
    -- e.g. a live-uploaded recording via the API. Training and evaluation
    use the MIT-BIH annotation's own sample index instead (see
    window_around_sample): that's ground truth, this is an approximation.

    min_distance_seconds=0.2 (300 bpm max) fixes a real gap found in the
    original project's peak detector, which used distance=0.7*fs (~86 bpm
    max) -- structurally blind to tachycardia, exactly the clinically
    important case an arrhythmia classifier most needs to catch.

    The amplitude threshold is relative to the signal's own mean/std
    (mean + amplitude_std_multiplier standard deviations), not a fixed
    absolute value, since raw filtered ECG amplitude varies by device/gain.

    An empty signal returns an empty array rather than computing
    mean/std of nothing: numpy doesn't raise for that, it emits
    "Mean of empty slice" / "invalid value encountered" RuntimeWarnings
    and returns NaN, which find_peaks would then silently receive as an
    unusable height threshold -- surfacing as warning-stream noise on
    every call for a truncated/zero-length record instead of a clear,
    explicit signal that nothing could be detected.
    """
    if len(signal) == 0:
        logger.warning("detect_r_peaks called with an empty signal; returning no peaks.")
        return np.array([], dtype=np.intp)

    height_threshold = np.mean(signal) + amplitude_std_multiplier * np.std(signal)
    min_distance_samples = int(min_distance_seconds * fs)
    peaks, _ = find_peaks(signal, distance=min_distance_samples, height=height_threshold)
    return peaks


def min_max_normalize(signal: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Scale a signal to [0, 1]. Returns zeros if the signal is flat.

    Uses a tolerance-based flatness check (not exact equality): a window
    that is flat except for floating-point noise on the order of 1e-14
    (plausible residue from upstream filtering of a genuinely flatlined
    segment) would otherwise pass the exact-equality check, get divided by
    that ~1e-14 range, and produce a spurious full-scale spike at a single
    sample -- indistinguishable from a real morphological feature to
    everything downstream.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if np.isclose(min_val, max_val):
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)
