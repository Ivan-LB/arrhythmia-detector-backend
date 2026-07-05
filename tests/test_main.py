from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from ecg_pipeline.labels import AAMI_CLASSES

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


FEATURE_COLUMNS = [
    "RPeakCount", "SpectralEnergy", "TotalPSD", "WaveletEnergy",
    "ShannonEntropy", "SignalSTD", "Skewness", "Kurtosis", "Variance",
]


def _read_bytes(relative_path: str) -> bytes:
    return (REPO_ROOT / relative_path).read_bytes()


def _train_tiny_model(tmp_path: Path) -> Path:
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


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return _train_tiny_model(tmp_path)


@pytest.fixture
def client(tmp_path: Path, model_dir: Path):
    app = create_app(model_dir=model_dir, records_dir=tmp_path / "records")
    with TestClient(app) as test_client:
        yield test_client


def _upload_record_230(client: TestClient, with_annotations: bool = False) -> str:
    files = {
        "hea_file": ("230.hea", _read_bytes("Data/Dataset/Train/230.hea")),
        "dat_file": ("230.dat", _read_bytes("Data/Dataset/Train/230.dat")),
    }
    if with_annotations:
        files["atr_file"] = ("230.atr", _read_bytes("Data/Dataset/Train/230.atr"))
    response = client.post("/records", files=files)
    return response.json()["record_id"]


class TestHealth:
    def test_reports_ok_with_model_version_and_aami_classes(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["aami_classes"] == list(AAMI_CLASSES)
        assert body["model_version"]


class TestUploadRecord:
    def test_uploads_a_real_record_and_returns_correct_metadata(self, client: TestClient):
        response = client.post(
            "/records",
            files={
                "hea_file": ("230.hea", _read_bytes("Data/Dataset/Train/230.hea")),
                "dat_file": ("230.dat", _read_bytes("Data/Dataset/Train/230.dat")),
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["sampling_rate"] == 360.0
        assert "MLII" in body["lead_names"]
        assert body["duration_seconds"] > 0
        assert body["record_id"]

    def test_rejects_mismatched_files_with_400(self, client: TestClient):
        response = client.post(
            "/records",
            files={
                "hea_file": ("230.hea", _read_bytes("Data/Dataset/Train/230.hea")),
                "dat_file": ("201.dat", _read_bytes("Data/Dataset/Train/201.dat")),
            },
        )
        assert response.status_code == 400

    def test_rejects_garbage_content_with_400_not_500(self, client: TestClient):
        response = client.post(
            "/records",
            files={
                "hea_file": ("fake.hea", b"not a real header"),
                "dat_file": ("fake.dat", b"not real data"),
            },
        )
        assert response.status_code == 400


class TestGetBeats:
    def test_returns_detected_beats_when_no_annotations_uploaded(self, client: TestClient):
        upload = client.post(
            "/records",
            files={
                "hea_file": ("230.hea", _read_bytes("Data/Dataset/Train/230.hea")),
                "dat_file": ("230.dat", _read_bytes("Data/Dataset/Train/230.dat")),
            },
        )
        record_id = upload.json()["record_id"]

        response = client.get(f"/records/{record_id}/beats")

        assert response.status_code == 200
        body = response.json()
        assert body["beat_source"] == "detected"
        assert len(body["beats"]) > 100  # record 230 has ~2000+ real beats
        for beat in body["beats"][:5]:
            assert beat["aami_class"] in AAMI_CLASSES
            assert 0.0 <= beat["confidence"] <= 1.0

    def test_returns_annotation_based_beats_when_atr_uploaded(self, client: TestClient):
        upload = client.post(
            "/records",
            files={
                "hea_file": ("230.hea", _read_bytes("Data/Dataset/Train/230.hea")),
                "dat_file": ("230.dat", _read_bytes("Data/Dataset/Train/230.dat")),
                "atr_file": ("230.atr", _read_bytes("Data/Dataset/Train/230.atr")),
            },
        )
        record_id = upload.json()["record_id"]

        response = client.get(f"/records/{record_id}/beats")

        assert response.status_code == 200
        assert response.json()["beat_source"] == "annotations"

    def test_returns_404_for_unknown_record_id(self, client: TestClient):
        response = client.get("/records/does-not-exist/beats")
        assert response.status_code == 404


class TestGetSignal:
    def test_returns_downsampled_signal(self, client: TestClient):
        upload = client.post(
            "/records",
            files={
                "hea_file": ("230.hea", _read_bytes("Data/Dataset/Train/230.hea")),
                "dat_file": ("230.dat", _read_bytes("Data/Dataset/Train/230.dat")),
            },
        )
        record_id = upload.json()["record_id"]

        response = client.get(f"/records/{record_id}/signal")

        assert response.status_code == 200
        body = response.json()
        assert body["sampling_rate"] == 360.0
        assert len(body["samples"]) <= 2100  # downsample target, with slack

    def test_returns_404_for_unknown_record_id(self, client: TestClient):
        response = client.get("/records/does-not-exist/signal")
        assert response.status_code == 404


@pytest.mark.anyio
class TestUploadSizeLimit:
    """Regression coverage for the memory-exhaustion fix: reading in
    bounded chunks with an early abort, instead of buffering the whole
    upload before checking its size.
    """

    async def test_read_upload_within_limit_aborts_before_reading_past_max_bytes(self):
        import io

        from fastapi import UploadFile

        from api.main import _read_upload_within_limit
        from api.records import InvalidRecordUpload

        oversized_content = b"x" * 1000
        upload = UploadFile(file=io.BytesIO(oversized_content), filename="big.dat")

        with pytest.raises(InvalidRecordUpload, match="exceeds the maximum upload size"):
            await _read_upload_within_limit(upload, "dat file", max_bytes=100)

    async def test_read_upload_within_limit_returns_full_content_when_under_the_limit(self):
        import io

        from fastapi import UploadFile

        from api.main import _read_upload_within_limit

        content = b"small content"
        upload = UploadFile(file=io.BytesIO(content), filename="small.dat")

        result = await _read_upload_within_limit(upload, "dat file", max_bytes=1000)

        assert result == content


class TestCaching:
    def test_repeated_beats_requests_only_compute_once(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        import api.main as main_module

        call_count = 0
        original = main_module.diagnose_beats

        def _counting_diagnose_beats(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(main_module, "diagnose_beats", _counting_diagnose_beats)

        record_id = _upload_record_230(client)
        first = client.get(f"/records/{record_id}/beats")
        second = client.get(f"/records/{record_id}/beats")

        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert call_count == 1

    def test_repeated_signal_requests_only_read_the_record_once(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        import api.main as main_module

        call_count = 0
        original = main_module.wfdb.rdrecord

        def _counting_rdrecord(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        record_id = _upload_record_230(client)
        monkeypatch.setattr(main_module.wfdb, "rdrecord", _counting_rdrecord)

        first = client.get(f"/records/{record_id}/signal")
        second = client.get(f"/records/{record_id}/signal")

        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert call_count == 1


class TestEviction:
    def test_oldest_record_is_evicted_once_capacity_is_exceeded(self, tmp_path: Path, model_dir: Path):
        app = create_app(model_dir=model_dir, records_dir=tmp_path / "records", max_stored_records=2)
        with TestClient(app) as client:
            first_id = _upload_record_230(client)
            second_id = _upload_record_230(client)
            third_id = _upload_record_230(client)

            assert app.state.records.keys() == {second_id, third_id}
            # the evicted record's on-disk directory must be removed too, not just the dict entry
            assert client.get(f"/records/{first_id}/beats").status_code == 404

    def test_records_within_capacity_are_not_evicted(self, tmp_path: Path, model_dir: Path):
        app = create_app(model_dir=model_dir, records_dir=tmp_path / "records", max_stored_records=5)
        with TestClient(app) as client:
            ids = [_upload_record_230(client) for _ in range(3)]
            assert set(app.state.records.keys()) == set(ids)


class TestGetEndpointsHandleUnexpectedFailuresCleanly:
    def test_get_beats_returns_a_clean_500_not_an_unhandled_crash_if_the_file_disappears(
        self, tmp_path: Path, model_dir: Path
    ):
        app = create_app(model_dir=model_dir, records_dir=tmp_path / "records")
        with TestClient(app, raise_server_exceptions=False) as client:
            record_id = _upload_record_230(client)
            stored = app.state.records[record_id].stored
            (stored.record_path.parent / f"{stored.record_path.name}.dat").unlink()

            response = client.get(f"/records/{record_id}/beats")

            assert response.status_code == 500
            assert response.json()["detail"] == f"Failed to process record {record_id!r}"
