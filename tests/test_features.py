import numpy as np
import pytest
from scipy.signal import periodogram
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

from ecg_pipeline import preprocessing
from ecg_pipeline.features import FEATURE_NAMES, BeatFeatures, extract_features

FS = 360.0
WINDOW_LENGTH = 360  # 1 second at 360 Hz, matching the project's window width


def _synthetic_window(rng: np.random.Generator) -> np.ndarray:
    """A small, already-normalized-like window for feature sanity checks."""
    t = np.arange(WINDOW_LENGTH) / FS
    signal = 0.5 + 0.1 * np.sin(2 * np.pi * 10.0 * t) + 0.01 * rng.standard_normal(WINDOW_LENGTH)
    return signal


class TestBeatFeaturesShape:
    def test_to_array_has_9_elements_in_feature_names_order(self):
        rng = np.random.default_rng(0)
        features = extract_features(_synthetic_window(rng), FS)
        array = features.to_array()
        assert array.shape == (9,)
        assert len(FEATURE_NAMES) == 9

    def test_to_dict_keys_match_feature_names(self):
        rng = np.random.default_rng(0)
        features = extract_features(_synthetic_window(rng), FS)
        assert tuple(features.to_dict().keys()) == FEATURE_NAMES


class TestStatisticalFeaturesMatchReferenceImplementation:
    def test_signal_std_variance_skewness_kurtosis(self):
        rng = np.random.default_rng(42)
        window = _synthetic_window(rng)
        features = extract_features(window, FS)

        assert features.signal_std == pytest.approx(float(np.std(window)))
        assert features.variance == pytest.approx(float(np.var(window)))
        assert features.skewness == pytest.approx(float(scipy_skew(window)))
        assert features.kurtosis == pytest.approx(float(scipy_kurtosis(window)))


class TestSpectralFeaturesAreHannWindowed:
    def test_spectral_energy_matches_a_manually_hann_windowed_fft(self):
        # This is the fix this rebuild makes: FFT must run on a Hann-windowed
        # segment, not a raw truncated one, to control spectral leakage.
        rng = np.random.default_rng(1)
        window = _synthetic_window(rng)
        features = extract_features(window, FS)

        hann_windowed = window * np.hanning(len(window))
        expected_fft = np.fft.rfft(hann_windowed)
        expected_spectral_energy = float(np.sum(np.abs(expected_fft) ** 2))

        assert features.spectral_energy == pytest.approx(expected_spectral_energy)

    def test_spectral_energy_does_not_match_a_raw_unwindowed_fft(self):
        # Guards against silently reverting to the old, leaky behavior.
        rng = np.random.default_rng(1)
        window = _synthetic_window(rng)
        features = extract_features(window, FS)

        raw_fft = np.fft.rfft(window)
        raw_spectral_energy = float(np.sum(np.abs(raw_fft) ** 2))

        assert features.spectral_energy != pytest.approx(raw_spectral_energy)

    def test_total_psd_matches_scipys_own_density_scaled_periodogram(self):
        # Guards against reverting to the old hand-rolled |X|^2/(fs/N)
        # formula, which was off from the standard |X|^2/(fs*N) density
        # formula by a factor of N^2.
        rng = np.random.default_rng(1)
        window = _synthetic_window(rng)
        features = extract_features(window, FS)

        _, expected_psd = periodogram(
            window, fs=FS, window="hann", detrend=False, scaling="density"
        )
        expected_total_psd = float(np.sum(expected_psd))

        assert features.total_psd == pytest.approx(expected_total_psd)


class TestWaveletFeaturesAreWellFormed:
    def test_wavelet_energy_is_positive_and_finite(self):
        rng = np.random.default_rng(2)
        features = extract_features(_synthetic_window(rng), FS)
        assert features.wavelet_energy > 0
        assert np.isfinite(features.wavelet_energy)

    def test_shannon_entropy_is_finite(self):
        rng = np.random.default_rng(2)
        features = extract_features(_synthetic_window(rng), FS)
        assert np.isfinite(features.shannon_entropy)

    def test_deterministic_for_the_same_input(self):
        rng = np.random.default_rng(3)
        window = _synthetic_window(rng)
        first = extract_features(window, FS)
        second = extract_features(window, FS)
        np.testing.assert_array_equal(first.to_array(), second.to_array())


class TestRPeakCount:
    def test_counts_clearly_separated_peaks_above_threshold(self):
        window = np.full(WINDOW_LENGTH, 0.2)
        window[100] = 0.9
        window[300] = 0.9  # 200 samples apart, > 0.45*360=162 min distance

        features = extract_features(window, FS)
        assert features.r_peak_count == 2

    def test_ignores_peaks_below_the_amplitude_threshold(self):
        window = np.full(WINDOW_LENGTH, 0.2)
        window[100] = 0.5  # below the 0.6 threshold
        window[300] = 0.5

        features = extract_features(window, FS)
        assert features.r_peak_count == 0

    def test_no_nan_features_for_sparse_signals_with_exact_zero_coefficients(self):
        # A mostly-flat window with sharp spikes produces exact-zero wavelet
        # coefficients at some scale/position combinations, which previously
        # produced NaN via 0 * log2(0) instead of treating it as 0 per the
        # Shannon entropy convention.
        window = np.full(WINDOW_LENGTH, 0.2)
        window[100] = 0.9
        window[300] = 0.9

        features = extract_features(window, FS)

        assert np.isfinite(features.shannon_entropy)
        assert not np.isnan(features.shannon_entropy)


class TestDegenerateWindowIsRejectedExplicitly:
    """A window with no real signal variation (e.g. a flatlined/disconnected
    lead segment) can't yield meaningful features. extract_features must
    raise clearly rather than let skew/kurtosis silently become NaN, or a
    wavelet-scale row with all-zero coefficients silently divide 0/0.
    """

    def test_raises_for_a_perfectly_flat_window(self):
        with pytest.raises(ValueError, match="no signal variation"):
            extract_features(np.full(WINDOW_LENGTH, 0.5), FS)

    def test_raises_for_an_all_zero_window(self):
        with pytest.raises(ValueError, match="no signal variation"):
            extract_features(np.zeros(WINDOW_LENGTH), FS)

    def test_integration_with_preprocessing_min_max_normalize(self):
        # preprocessing.min_max_normalize documents and tests that a flat
        # input signal produces an all-zero output -- confirm that feeding
        # that documented output straight into extract_features raises
        # clearly rather than silently producing NaN, closing the exact gap
        # between the two modules' otherwise-fully-covered unit tests.
        flat_signal = np.full(WINDOW_LENGTH, 7.0)
        normalized = preprocessing.min_max_normalize(flat_signal)

        with pytest.raises(ValueError, match="no signal variation"):
            extract_features(normalized, FS)
