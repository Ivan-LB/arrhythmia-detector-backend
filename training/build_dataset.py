"""Build the DS1/DS2 feature datasets from raw MIT-BIH records.

Two layers, kept deliberately separate:
- extract_beats_from_record: pure signal-in, rows-out logic, unit-tested
  with synthetic signals (no wfdb/file dependency).
- build_dataset: the I/O glue that reads real records via wfdb and writes
  the resulting CSVs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import wfdb

from ecg_pipeline import features as features_module
from ecg_pipeline import labels, preprocessing, splits

logger = logging.getLogger(__name__)

DATASET_COLUMNS: tuple[str, ...] = (*features_module.FEATURE_NAMES, "AAMIClass", "RecordID")

SkipCounts = dict[str, int]
_EMPTY_SKIP_COUNTS: SkipCounts = {
    "non_beat_symbol": 0,
    "window_out_of_bounds": 0,
    "degenerate_window": 0,
}


def extract_beats_from_record(
    signal: npt.NDArray[np.float64],
    fs: float,
    ann_samples: npt.NDArray[np.int64],
    ann_symbols: list[str],
    record_id: int,
) -> tuple[list[dict], SkipCounts]:
    """Extract labeled feature rows from one MLII channel and its beat annotations.

    Returns (rows, skip_counts). Annotations the AAMI mapping doesn't cover
    (non-beat markers), windows that run off the recording's edge, and
    degenerate (flatlined) windows are all counted and skipped rather than
    silently dropped or fabricated a value for -- see
    docs/data-pipeline-architecture.md sec 5 and sec 11.
    """
    filtered = preprocessing.apply_notch_filter(preprocessing.remove_baseline(signal), fs)

    rows: list[dict] = []
    skip_counts: SkipCounts = dict(_EMPTY_SKIP_COUNTS)

    for sample, symbol in zip(ann_samples, ann_symbols):
        aami_class = labels.to_aami_class(symbol)
        if aami_class is None:
            skip_counts["non_beat_symbol"] += 1
            continue

        window = preprocessing.window_around_sample(filtered, int(sample), fs)
        if window is None:
            skip_counts["window_out_of_bounds"] += 1
            continue

        normalized = preprocessing.min_max_normalize(window)
        try:
            beat_features = features_module.extract_features(normalized, fs)
        except ValueError:
            skip_counts["degenerate_window"] += 1
            continue

        rows.append({**beat_features.to_dict(), "AAMIClass": aami_class, "RecordID": record_id})

    return rows, skip_counts


def _record_directory(record_id: int, dataset_root: Path) -> Path:
    for candidate in ("Train", "Test"):
        folder = dataset_root / candidate
        if (folder / f"{record_id}.hea").exists():
            return folder
    raise FileNotFoundError(f"Record {record_id} not found under {dataset_root}")


def build_dataset(
    dataset_root: Path,
    output_dir: Path,
    record_ids: tuple[int, ...] | None = None,
) -> None:
    """Read raw MIT-BIH records and write dataset_ds1.csv / dataset_ds2.csv.

    record_ids defaults to the full DS1+DS2 set; callers (tests) can pass a
    small subset for a fast end-to-end check against real files.
    """
    all_records = record_ids if record_ids is not None else splits.DS1 + splits.DS2
    ds1_rows: list[dict] = []
    ds2_rows: list[dict] = []
    total_skips: SkipCounts = {**_EMPTY_SKIP_COUNTS, "no_mlii_channel": 0}

    for record_id in all_records:
        folder = _record_directory(record_id, dataset_root)
        record_path = str(folder / str(record_id))

        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, "atr")

        try:
            mlii_index = preprocessing.find_mlii_channel(record.sig_name)
        except ValueError:
            logger.warning("Record %s has no MLII channel; skipping entirely.", record_id)
            total_skips["no_mlii_channel"] += 1
            continue

        signal = record.p_signal[:, mlii_index]
        rows, skip_counts = extract_beats_from_record(
            signal, record.fs, annotation.sample, annotation.symbol, record_id
        )
        for key, value in skip_counts.items():
            total_skips[key] += value

        split_name = splits.split_for_record(record_id)
        (ds1_rows if split_name == "DS1" else ds2_rows).extend(rows)

        logger.info("Record %s (%s): %d beats extracted", record_id, split_name, len(rows))

    logger.info("Extraction complete. Skip counts: %s", total_skips)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Always pass columns=DATASET_COLUMNS explicitly: an empty rows list (e.g.
    # a DS1-only record subset in a test) would otherwise produce a DataFrame
    # with zero columns, writing a headerless CSV that pandas can't even read
    # back (EmptyDataError), instead of a valid, zero-row, correctly-shaped one.
    pd.DataFrame(ds1_rows, columns=DATASET_COLUMNS).to_csv(output_dir / "dataset_ds1.csv", index=False)
    pd.DataFrame(ds2_rows, columns=DATASET_COLUMNS).to_csv(output_dir / "dataset_ds2.csv", index=False)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Data/Dataset")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("Data")
    build_dataset(dataset_root, output_dir)
