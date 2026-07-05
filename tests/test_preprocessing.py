import numpy as np
import pytest

from ecg_pipeline import preprocessing


class TestFindMliiChannel:
    def test_finds_mlii_among_other_leads(self):
        assert preprocessing.find_mlii_channel(["V5", "MLII"]) == 1
        assert preprocessing.find_mlii_channel(["MLII", "V1"]) == 0

    def test_raises_when_mlii_is_absent(self):
        with pytest.raises(ValueError, match="MLII"):
            preprocessing.find_mlii_channel(["V1", "V5"])


class TestRemoveBaseline:
    def test_removes_dc_offset(self):
        signal = np.array([5.0, 6.0, 7.0, 6.0, 5.0]) + 100.0
        centered = preprocessing.remove_baseline(signal)
        assert np.isclose(np.mean(centered), 0.0)

    def test_preserves_shape_around_the_mean(self):
        signal = np.array([1.0, 2.0, 3.0]) + 50.0
        centered = preprocessing.remove_baseline(signal)
        np.testing.assert_allclose(centered, [-1.0, 0.0, 1.0])


class TestApplyNotchFilter:
    def test_attenuates_60hz_powerline_interference(self):
        fs = 360.0
        duration_seconds = 2.0
        t = np.linspace(0, duration_seconds, int(fs * duration_seconds), endpoint=False)
        pure_60hz = np.sin(2 * np.pi * 60.0 * t)

        filtered = preprocessing.apply_notch_filter(pure_60hz, fs)

        # a working 60 Hz notch should crush a pure 60 Hz tone's energy
        original_energy = np.sum(pure_60hz**2)
        filtered_energy = np.sum(filtered**2)
        assert filtered_energy < 0.1 * original_energy

    def test_preserves_a_signal_far_from_60hz(self):
        fs = 360.0
        duration_seconds = 2.0
        t = np.linspace(0, duration_seconds, int(fs * duration_seconds), endpoint=False)
        low_freq_signal = np.sin(2 * np.pi * 5.0 * t)

        filtered = preprocessing.apply_notch_filter(low_freq_signal, fs)

        original_energy = np.sum(low_freq_signal**2)
        filtered_energy = np.sum(filtered**2)
        assert filtered_energy > 0.8 * original_energy

    def test_is_zero_phase(self):
        # filtfilt (zero-phase) is required, not lfilter (causal): beat
        # windows are centered on the annotation's own sample index, so any
        # phase delay would shift signal content relative to that center.
        fs = 360.0
        signal = np.zeros(200)
        signal[100] = 1.0  # an impulse at a known, exact sample

        filtered = preprocessing.apply_notch_filter(signal, fs)

        assert np.argmax(np.abs(filtered)) == 100

    def test_raises_for_fs_at_or_below_twice_the_notch_frequency(self):
        with pytest.raises(ValueError, match="fs"):
            preprocessing.apply_notch_filter(np.zeros(10), fs=100.0)


class TestWindowAroundSample:
    def test_returns_window_of_expected_length(self):
        fs = 360.0
        signal = np.arange(2000, dtype=float)
        window = preprocessing.window_around_sample(
            signal, center_sample=1000, fs=fs, width_seconds=1.0
        )
        assert window is not None
        assert len(window) == int(fs)

    def test_window_is_centered_on_the_given_sample(self):
        fs = 360.0
        signal = np.arange(2000, dtype=float)
        window = preprocessing.window_around_sample(
            signal, center_sample=1000, fs=fs, width_seconds=1.0
        )
        half_width = int(fs / 2)
        np.testing.assert_array_equal(window, signal[1000 - half_width : 1000 + half_width])

    def test_returns_none_when_window_runs_off_the_start(self):
        fs = 360.0
        signal = np.arange(2000, dtype=float)
        window = preprocessing.window_around_sample(
            signal, center_sample=10, fs=fs, width_seconds=1.0
        )
        assert window is None

    def test_returns_none_when_window_runs_off_the_end(self):
        fs = 360.0
        signal = np.arange(2000, dtype=float)
        window = preprocessing.window_around_sample(
            signal, center_sample=1995, fs=fs, width_seconds=1.0
        )
        assert window is None

    def test_raises_for_non_positive_width_seconds(self):
        signal = np.arange(2000, dtype=float)
        with pytest.raises(ValueError, match="width_seconds"):
            preprocessing.window_around_sample(signal, 1000, fs=360.0, width_seconds=0)

    def test_raises_for_non_positive_fs(self):
        signal = np.arange(2000, dtype=float)
        with pytest.raises(ValueError, match="fs"):
            preprocessing.window_around_sample(signal, 1000, fs=0.0, width_seconds=1.0)


class TestMinMaxNormalize:
    def test_scales_to_zero_one_range(self):
        signal = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        normalized = preprocessing.min_max_normalize(signal)
        assert np.isclose(np.min(normalized), 0.0)
        assert np.isclose(np.max(normalized), 1.0)

    def test_flat_signal_returns_zeros_not_nan(self):
        signal = np.full(10, 3.0)
        normalized = preprocessing.min_max_normalize(signal)
        np.testing.assert_array_equal(normalized, np.zeros(10))

    def test_near_flat_signal_within_floating_point_noise_also_returns_zeros(self):
        # Guards against the exact-equality bug: a signal that's flat except
        # for ~1e-14 floating-point residue (plausible upstream filter
        # artifact on a genuinely flatlined segment) must not be treated as
        # having real variation -- dividing by a ~1e-14 range would produce
        # a spurious full-scale spike at a single sample.
        signal = np.full(10, 0.5)
        signal[3] += 1e-14
        normalized = preprocessing.min_max_normalize(signal)
        np.testing.assert_array_equal(normalized, np.zeros(10))
