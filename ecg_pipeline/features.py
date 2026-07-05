"""9-feature extraction from a normalized, beat-centered ECG window.

Expects the window to already be min-max normalized (see
preprocessing.min_max_normalize) -- this module only computes features from
it, keeping preprocessing and feature engineering as separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pywt
from scipy.signal import find_peaks, periodogram
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

SECONDARY_PEAK_MIN_DISTANCE_SECONDS = 0.45
SECONDARY_PEAK_AMPLITUDE_THRESHOLD = 0.6
WAVELET_SCALES = np.arange(1, 17)
WAVELET_NAME = "mexh"

FEATURE_NAMES: tuple[str, ...] = (
    "RPeakCount",
    "SpectralEnergy",
    "TotalPSD",
    "WaveletEnergy",
    "ShannonEntropy",
    "SignalSTD",
    "Skewness",
    "Kurtosis",
    "Variance",
)


@dataclass(frozen=True)
class BeatFeatures:
    r_peak_count: int
    spectral_energy: float
    total_psd: float
    wavelet_energy: float
    shannon_entropy: float
    signal_std: float
    skewness: float
    kurtosis: float
    variance: float

    def to_array(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                self.r_peak_count,
                self.spectral_energy,
                self.total_psd,
                self.wavelet_energy,
                self.shannon_entropy,
                self.signal_std,
                self.skewness,
                self.kurtosis,
                self.variance,
            ]
        )

    def to_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.to_array()))


def _require_non_degenerate(window: npt.NDArray[np.float64]) -> None:
    """A window with no real signal variation (e.g. a flatlined/disconnected
    lead segment) can't yield meaningful shape/spectral features. Raise
    clearly instead of silently returning NaN (skew/kurtosis) or a fabricated
    placeholder value -- the caller (dataset builder) must exclude this beat,
    not train on an invented number for it.
    """
    if np.isclose(np.min(window), np.max(window)):
        raise ValueError(
            "Window has no signal variation (min == max within floating point tolerance); "
            "cannot compute shape/spectral features. Exclude this beat from the dataset "
            "rather than substituting a placeholder value."
        )


def _count_secondary_peaks(window: npt.NDArray[np.float64], fs: float) -> int:
    min_distance = int(SECONDARY_PEAK_MIN_DISTANCE_SECONDS * fs)
    peak_indices, _ = find_peaks(window, distance=min_distance)
    peak_indices = peak_indices[window[peak_indices] > SECONDARY_PEAK_AMPLITUDE_THRESHOLD]
    return len(peak_indices)


def _spectral_features(window: npt.NDArray[np.float64], fs: float) -> tuple[float, float]:
    """Hann-windowed FFT energy, and a standard-scaling power spectral density.

    Both are computed on a Hann-windowed copy of the segment to control
    spectral leakage (see docs/data-pipeline-architecture.md sec 5). TotalPSD
    is delegated to scipy.signal.periodogram's "density" scaling rather than
    hand-rolled: the textbook periodogram-density formula is |X[k]|^2 /
    (fs * N), not |X[k]|^2 / (fs / N) -- an earlier hand-rolled version used
    the latter, off by a factor of N^2, and its manual one-sided-spectrum
    doubling was only valid for even-length windows. Delegating gets both
    the scaling and the even/odd-length handling right for free.
    """
    hann_windowed = window * np.hanning(len(window))
    fft_coeffs = np.fft.rfft(hann_windowed)
    spectral_energy = float(np.sum(np.abs(fft_coeffs) ** 2))

    # detrend=False: baseline removal already happened upstream
    # (preprocessing.remove_baseline); don't detrend twice, implicitly.
    _, psd = periodogram(window, fs=fs, window="hann", detrend=False, scaling="density")
    total_psd = float(np.sum(psd))

    return spectral_energy, total_psd


def _wavelet_features(window: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Continuous wavelet transform energy and mean Shannon entropy across scales.

    CWT doesn't carry FFT's periodicity assumption, so it doesn't need the
    same windowing treatment as the spectral features above.
    """
    coefficients, _ = pywt.cwt(window, WAVELET_SCALES, WAVELET_NAME)
    wavelet_energy = float(np.sum(coefficients**2))

    energy_per_scale = np.sum(coefficients**2, axis=1)
    energy_per_scale[energy_per_scale == 0] = np.finfo(float).eps
    probability = coefficients**2 / energy_per_scale[:, None]

    # Guard the second normalization the same way as the first: if an entire
    # wavelet scale row was all-zero (a fully degenerate signal at that
    # scale), its row-sum after the first division is 0, and dividing by
    # that would be 0/0 = NaN rather than a genuine zero-drift correction.
    probability_row_sums = np.sum(probability, axis=1)
    probability_row_sums[probability_row_sums == 0] = np.finfo(float).eps
    probability /= probability_row_sums[:, None]

    # 0 * log2(0) is conventionally 0 for Shannon entropy, but numpy evaluates
    # it as 0 * -inf = NaN. Compute log2 only where probability > 0 so exact
    # zeros (common for sparse/spiky windows) don't poison the feature.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probability = np.where(probability > 0, np.log2(probability), 0.0)
    entropy_per_scale = -np.sum(probability * log_probability, axis=1)

    return wavelet_energy, float(np.mean(entropy_per_scale))


def extract_features(window: npt.NDArray[np.float64], fs: float) -> BeatFeatures:
    """Extract the 9 engineered features from a normalized, beat-centered window.

    Raises ValueError if the window has no real signal variation -- see
    _require_non_degenerate.
    """
    _require_non_degenerate(window)

    spectral_energy, total_psd = _spectral_features(window, fs)
    wavelet_energy, shannon_entropy = _wavelet_features(window)

    return BeatFeatures(
        r_peak_count=_count_secondary_peaks(window, fs),
        spectral_energy=spectral_energy,
        total_psd=total_psd,
        wavelet_energy=wavelet_energy,
        shannon_entropy=shannon_entropy,
        signal_std=float(np.std(window)),
        skewness=float(scipy_skew(window)),
        kurtosis=float(scipy_kurtosis(window)),
        variance=float(np.var(window)),
    )
