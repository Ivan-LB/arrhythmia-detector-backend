from pathlib import Path

import numpy as np
import pytest
import wfdb

from api.records import InvalidRecordUpload, save_uploaded_record

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_bytes(relative_path: str) -> bytes:
    return (REPO_ROOT / relative_path).read_bytes()


def _write_synthetic_record(write_dir: Path, record_name: str, sig_name: list[str]) -> tuple[bytes, bytes]:
    """Write a tiny, real, valid wfdb record with a caller-chosen lead name
    -- used to test behavior for records that don't have MLII, which none
    of the real fixture files in this repo exercise.
    """
    signal = np.zeros((360, len(sig_name)))
    wfdb.wrsamp(
        record_name,
        fs=360,
        units=["mV"] * len(sig_name),
        sig_name=sig_name,
        p_signal=signal,
        fmt=["16"] * len(sig_name),
        write_dir=str(write_dir),
    )
    hea_bytes = (write_dir / f"{record_name}.hea").read_bytes()
    dat_bytes = (write_dir / f"{record_name}.dat").read_bytes()
    return hea_bytes, dat_bytes


class TestSaveUploadedRecord:
    def test_saves_a_valid_hea_dat_pair_and_returns_a_readable_record(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")

        stored = save_uploaded_record(
            records_root=tmp_path,
            hea_filename="230.hea",
            hea_content=hea_bytes,
            dat_filename="230.dat",
            dat_content=dat_bytes,
        )

        record = wfdb.rdrecord(str(stored.record_path))
        assert record.fs == 360
        assert "MLII" in record.sig_name

    def test_generates_a_fresh_record_id_not_client_controlled(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")

        first = save_uploaded_record(tmp_path, "230.hea", hea_bytes, "230.dat", dat_bytes)
        second = save_uploaded_record(tmp_path, "230.hea", hea_bytes, "230.dat", dat_bytes)

        assert first.record_id != second.record_id

    def test_rejects_mismatched_base_filenames(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")

        with pytest.raises(InvalidRecordUpload, match="base name"):
            save_uploaded_record(tmp_path, "230.hea", hea_bytes, "201.dat", dat_bytes)

    def test_rejects_path_traversal_in_hea_filename(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")

        with pytest.raises(InvalidRecordUpload):
            save_uploaded_record(
                tmp_path, "../../../../etc/230.hea", hea_bytes, "../../../../etc/230.dat", dat_bytes
            )

    def test_rejects_path_traversal_via_absolute_path_filename(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")

        with pytest.raises(InvalidRecordUpload):
            save_uploaded_record(tmp_path, "/etc/230.hea", hea_bytes, "/etc/230.dat", dat_bytes)

    def test_rejects_a_dat_file_over_the_size_limit(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        oversized_dat = b"0" * (200 * 1024 * 1024 + 1)  # over the 200MB cap

        with pytest.raises(InvalidRecordUpload, match="size"):
            save_uploaded_record(tmp_path, "230.hea", hea_bytes, "230.dat", oversized_dat)

    def test_rejects_content_that_is_not_a_valid_wfdb_record(self, tmp_path: Path):
        with pytest.raises(InvalidRecordUpload):
            save_uploaded_record(tmp_path, "fake.hea", b"not a real header", "fake.dat", b"not real data either")

    def test_each_upload_is_isolated_in_its_own_directory(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")

        first = save_uploaded_record(tmp_path, "230.hea", hea_bytes, "230.dat", dat_bytes)
        second = save_uploaded_record(tmp_path, "230.hea", hea_bytes, "230.dat", dat_bytes)

        assert first.record_path.parent != second.record_path.parent


class TestSaveUploadedRecordWithAnnotations:
    def test_accepts_a_matching_atr_file_and_reports_annotations_present(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")
        atr_bytes = _read_bytes("Data/Dataset/Train/230.atr")

        stored = save_uploaded_record(
            tmp_path, "230.hea", hea_bytes, "230.dat", dat_bytes, atr_filename="230.atr", atr_content=atr_bytes
        )

        assert stored.has_annotations is True
        annotation = wfdb.rdann(str(stored.record_path), "atr")
        assert len(annotation.sample) > 0

    def test_rejects_atr_filename_provided_without_atr_content(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")

        with pytest.raises(InvalidRecordUpload, match="must both be provided"):
            save_uploaded_record(
                tmp_path, "230.hea", hea_bytes, "230.dat", dat_bytes, atr_filename="230.atr", atr_content=None
            )

    def test_without_an_atr_file_has_annotations_is_false(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")

        stored = save_uploaded_record(tmp_path, "230.hea", hea_bytes, "230.dat", dat_bytes)

        assert stored.has_annotations is False

    def test_rejects_an_atr_file_with_a_mismatched_base_name(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")
        atr_bytes = _read_bytes("Data/Dataset/Train/201.atr")

        with pytest.raises(InvalidRecordUpload, match="base name"):
            save_uploaded_record(
                tmp_path, "230.hea", hea_bytes, "230.dat", dat_bytes, atr_filename="201.atr", atr_content=atr_bytes
            )

    def test_rejects_path_traversal_in_atr_filename(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")
        atr_bytes = _read_bytes("Data/Dataset/Train/230.atr")

        with pytest.raises(InvalidRecordUpload):
            save_uploaded_record(
                tmp_path,
                "230.hea",
                hea_bytes,
                "230.dat",
                dat_bytes,
                atr_filename="../../../etc/230.atr",
                atr_content=atr_bytes,
            )


class TestSaveUploadedRecordSecurityFixes:
    def test_rejects_a_nul_byte_in_the_hea_filename_with_a_clean_error_not_a_crash(self, tmp_path: Path):
        # Regression test: Path() doesn't reject an embedded NUL byte at
        # construction time, so this used to pass filename validation and
        # only fail later at the OS syscall boundary in write_bytes() with
        # an unhandled ValueError -- a real crash, not a clean 400.
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")

        with pytest.raises(InvalidRecordUpload, match="control characters"):
            save_uploaded_record(tmp_path, "230.hea\x00.txt", hea_bytes, "230.dat", dat_bytes)

    def test_rejects_a_nul_byte_in_the_dat_filename(self, tmp_path: Path):
        hea_bytes = _read_bytes("Data/Dataset/Train/230.hea")
        dat_bytes = _read_bytes("Data/Dataset/Train/230.dat")

        with pytest.raises(InvalidRecordUpload, match="control characters"):
            save_uploaded_record(tmp_path, "230.hea", hea_bytes, "230.dat\x00.txt", dat_bytes)

    def test_rejects_a_record_with_no_mlii_lead_at_upload_time_not_later(self, tmp_path: Path):
        # Regression test: a record without MLII used to be accepted at
        # upload (200), then crash every subsequent GET /beats and
        # GET /signal call with an unhandled ValueError from
        # find_mlii_channel deep in ecg_pipeline.preprocessing. Catching
        # this at upload time means the client gets one clear 400 instead
        # of a misleading "success" followed by every read failing.
        write_dir = tmp_path / "synthetic_source"
        write_dir.mkdir()
        hea_bytes, dat_bytes = _write_synthetic_record(write_dir, "novel", sig_name=["V1", "V5"])

        with pytest.raises(InvalidRecordUpload):
            save_uploaded_record(tmp_path, "novel.hea", hea_bytes, "novel.dat", dat_bytes)
