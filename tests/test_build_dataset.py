from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.build_dataset import (
    DATASET_COLUMNS,
    _record_directory,
    build_dataset,
    extract_beats_from_record,
)

FS = 360.0
SIGNAL_LENGTH = 8000
FLAT_REGION_CENTER = 6000  # far enough from the sine region that filtfilt's
# transient decays to numerically-exact flatness (verified: at 1000 samples
# separation the residual is ~7e-8, just enough to fail np.isclose; at 5000+
# samples separation it's ~1e-38, i.e. exactly flat in any practical sense)


def _signal_with_one_real_beat_and_one_flat_region() -> np.ndarray:
    """Mostly flat signal with real sine variation only around sample 500.

    FLAT_REGION_CENTER's window never overlaps the sine region (320-680) and
    is far enough away that it stays flat/degenerate after preprocessing.
    """
    signal = np.full(SIGNAL_LENGTH, 0.5)
    t = np.arange(320, 680) / FS
    signal[320:680] += 0.2 * np.sin(2 * np.pi * 10.0 * t)
    return signal


class TestExtractBeatsFromRecord:
    def test_extracts_one_row_per_valid_beat(self):
        signal = _signal_with_one_real_beat_and_one_flat_region()
        # Only the real-variation beat at 500 is annotated here; keep the
        # flat region out of this case (covered separately below).
        rows, skip_counts = extract_beats_from_record(
            signal, FS, np.array([500]), ["N"], record_id=100
        )

        assert len(rows) == 1
        assert rows[0]["AAMIClass"] == "N"
        assert rows[0]["RecordID"] == 100
        assert set(skip_counts.values()) == {0}

    def test_multiple_valid_beats_all_extracted_with_correct_labels(self):
        signal = np.full(SIGNAL_LENGTH, 0.5)
        for center in (400, 900, 1400):
            t = np.arange(center - 180, center + 180) / FS
            signal[center - 180 : center + 180] += 0.2 * np.sin(2 * np.pi * 10.0 * t)

        rows, skip_counts = extract_beats_from_record(
            signal, FS, np.array([400, 900, 1400]), ["N", "V", "A"], record_id=200
        )

        assert len(rows) == 3
        assert [row["AAMIClass"] for row in rows] == ["N", "V", "S"]
        assert set(skip_counts.values()) == {0}

    def test_skips_and_counts_non_beat_annotation_symbols(self):
        signal = _signal_with_one_real_beat_and_one_flat_region()
        # "+" is a rhythm-change marker, not a beat symbol -- not in the AAMI mapping.
        rows, skip_counts = extract_beats_from_record(
            signal, FS, np.array([500]), ["+"], record_id=100
        )

        assert rows == []
        assert skip_counts["non_beat_symbol"] == 1

    def test_skips_and_counts_windows_that_run_off_the_recording_edge(self):
        signal = _signal_with_one_real_beat_and_one_flat_region()
        # Sample 10 is too close to the start for a full +-0.5s window.
        rows, skip_counts = extract_beats_from_record(
            signal, FS, np.array([10]), ["N"], record_id=100
        )

        assert rows == []
        assert skip_counts["window_out_of_bounds"] == 1

    def test_skips_and_counts_degenerate_flat_windows(self):
        signal = _signal_with_one_real_beat_and_one_flat_region()
        # FLAT_REGION_CENTER's window never overlaps the sine region -- stays flat.
        rows, skip_counts = extract_beats_from_record(
            signal, FS, np.array([FLAT_REGION_CENTER]), ["N"], record_id=100
        )

        assert rows == []
        assert skip_counts["degenerate_window"] == 1

    def test_mixed_batch_tallies_each_skip_reason_independently(self):
        signal = _signal_with_one_real_beat_and_one_flat_region()
        ann_samples = np.array([500, FLAT_REGION_CENTER, 10, 500])
        ann_symbols = ["N", "N", "N", "+"]

        rows, skip_counts = extract_beats_from_record(
            signal, FS, ann_samples, ann_symbols, record_id=100
        )

        assert len(rows) == 1  # only the first sample=500/"N" pair is valid
        assert skip_counts == {
            "non_beat_symbol": 1,
            "window_out_of_bounds": 1,
            "degenerate_window": 1,
        }


class TestRecordDirectory:
    def test_finds_record_in_train_folder(self, tmp_path: Path):
        (tmp_path / "Train").mkdir()
        (tmp_path / "Train" / "200.hea").touch()
        assert _record_directory(200, tmp_path) == tmp_path / "Train"

    def test_finds_record_in_test_folder(self, tmp_path: Path):
        (tmp_path / "Test").mkdir()
        (tmp_path / "Test" / "100.hea").touch()
        assert _record_directory(100, tmp_path) == tmp_path / "Test"

    def test_raises_when_record_not_found_anywhere(self, tmp_path: Path):
        (tmp_path / "Train").mkdir()
        (tmp_path / "Test").mkdir()
        with pytest.raises(FileNotFoundError, match="999"):
            _record_directory(999, tmp_path)


class TestBuildDatasetIntegration:
    """Exercises the real wfdb I/O path against one small, real MIT-BIH
    record, rather than only the synthetic unit tests above -- proves the
    wiring (record lookup, wfdb reading, MLII selection, CSV writing)
    actually works end to end, not just the pure extraction logic.
    """

    def test_builds_a_real_ds1_record_into_dataset_ds1_csv(self, tmp_path: Path):
        build_dataset(Path("Data/Dataset"), tmp_path, record_ids=(230,))

        ds1_csv = tmp_path / "dataset_ds1.csv"
        ds2_csv = tmp_path / "dataset_ds2.csv"
        assert ds1_csv.exists()
        assert ds2_csv.exists()

        df = pd.read_csv(ds1_csv)
        assert len(df) > 1000  # record 230 has ~2700 real annotated beats
        assert set(df["RecordID"]) == {230}
        assert set(df["AAMIClass"]).issubset({"N", "S", "V", "F", "Q"})
        assert not df.isna().any().any()

        # record 230 is DS1-only; nothing should land in the DS2 file, but
        # the file must still be a valid, readable, correctly-columned CSV --
        # not a headerless empty file pandas can't parse back.
        empty_ds2 = pd.read_csv(ds2_csv)
        assert len(empty_ds2) == 0
        assert list(empty_ds2.columns) == list(DATASET_COLUMNS)
