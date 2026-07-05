from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from ecg_pipeline.labels import AAMI_CLASSES

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_COLUMNS = [
    "RPeakCount", "SpectralEnergy", "TotalPSD", "WaveletEnergy",
    "ShannonEntropy", "SignalSTD", "Skewness", "Kurtosis", "Variance",
]


def _read_bytes(relative_path: str) -> bytes:
    return (REPO_ROOT / relative_path).read_bytes()


@pytest.fixture
def client(tmp_path: Path):
    from training.train import train

    rng = np.random.default_rng(0)
    rows = [
        {**{c: rng.standard_normal() for c in FEATURE_COLUMNS}, "AAMIClass": rng.choice(AAMI_CLASSES), "RecordID": rid}
        for rid in range(1, 11)
        for _ in range(30)
    ]
    ds1_csv = tmp_path / "dataset_ds1.csv"
    pd.DataFrame(rows).to_csv(ds1_csv, index=False)
    model_dir = train(ds1_csv, tmp_path / "models", epochs=1, batch_size=16, seed=42)

    app = create_app(model_dir=model_dir, records_dir=tmp_path / "records")
    with TestClient(app) as test_client:
        yield test_client


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
