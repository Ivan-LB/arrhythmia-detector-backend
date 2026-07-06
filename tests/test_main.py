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


class TestCors:
    """Regression coverage for the missing-CORS-middleware gap found while
    building the separate arrhythmia-detector-web frontend: a browser
    fetch() from that app's dev server (http://localhost:3000) failed with
    a CORS rejection even though curl/TestClient hit the same endpoints
    fine, since neither of those enforce the browser's same-origin policy.
    """

    def test_allows_the_configured_frontend_origin(self, client: TestClient):
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_preflight_allows_post_to_records_from_the_frontend_origin(self, client: TestClient):
        response = client.options(
            "/records",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_does_not_reflect_an_origin_outside_the_allowlist(self, client: TestClient):
        response = client.get("/health", headers={"Origin": "http://evil.example.com"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


class TestParseCorsAllowedOrigins:
    """_app_from_env() itself needs a real MODEL_DIR + trained model to
    exercise, so the parsing rule is covered directly against the
    extracted pure function instead.
    """

    def test_returns_the_default_when_the_env_var_is_unset(self):
        from api.main import DEFAULT_CORS_ALLOWED_ORIGINS, _parse_cors_allowed_origins

        assert _parse_cors_allowed_origins(None) == DEFAULT_CORS_ALLOWED_ORIGINS

    def test_splits_and_strips_a_comma_separated_list(self):
        from api.main import _parse_cors_allowed_origins

        result = _parse_cors_allowed_origins("http://localhost:3000, https://example.com ,http://a.test")
        assert result == ("http://localhost:3000", "https://example.com", "http://a.test")

    def test_falls_back_to_the_default_when_every_segment_is_blank(self, caplog: pytest.LogCaptureFixture):
        from api.main import DEFAULT_CORS_ALLOWED_ORIGINS, _parse_cors_allowed_origins

        with caplog.at_level("WARNING"):
            result = _parse_cors_allowed_origins(" , ,")

        assert result == DEFAULT_CORS_ALLOWED_ORIGINS
        assert "no usable origins" in caplog.text


class TestParseNumericEnv:
    def test_returns_the_default_when_the_value_is_none(self):
        from api.main import _parse_numeric_env

        assert _parse_numeric_env("X", None, 42, int) == 42
        assert _parse_numeric_env("Y", None, 60.0, float) == 60.0

    def test_casts_a_valid_value(self):
        from api.main import _parse_numeric_env

        assert _parse_numeric_env("X", "7", 42, int) == 7
        assert _parse_numeric_env("Y", "1.5", 60.0, float) == 1.5

    def test_raises_a_runtime_error_naming_the_variable_and_bad_value_on_invalid_input(self):
        from api.main import _parse_numeric_env

        with pytest.raises(RuntimeError, match="RATE_LIMIT_MAX_REQUESTS.*not-a-number"):
            _parse_numeric_env("RATE_LIMIT_MAX_REQUESTS", "not-a-number", 10, int)


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


class TestUploadRateLimiter:
    """Direct unit tests against the sliding-window counter itself -- `now`
    is an explicit parameter, not read from a real clock, so the window
    can be exercised deterministically without sleeping in tests.
    """

    def test_allows_requests_up_to_the_configured_limit(self):
        from api.main import _UploadRateLimiter

        limiter = _UploadRateLimiter(max_requests=3, window_seconds=60, max_tracked_clients=100)

        assert limiter.check("1.2.3.4", now=0.0) is None
        assert limiter.check("1.2.3.4", now=1.0) is None
        assert limiter.check("1.2.3.4", now=2.0) is None

    def test_rejects_the_request_that_exceeds_the_limit(self):
        from api.main import _UploadRateLimiter

        limiter = _UploadRateLimiter(max_requests=2, window_seconds=60, max_tracked_clients=100)
        limiter.check("1.2.3.4", now=0.0)
        limiter.check("1.2.3.4", now=1.0)

        retry_after = limiter.check("1.2.3.4", now=2.0)

        assert retry_after is not None
        assert retry_after > 0

    def test_tracks_different_clients_independently(self):
        from api.main import _UploadRateLimiter

        limiter = _UploadRateLimiter(max_requests=1, window_seconds=60, max_tracked_clients=100)

        assert limiter.check("1.2.3.4", now=0.0) is None
        assert limiter.check("5.6.7.8", now=0.0) is None
        assert limiter.check("1.2.3.4", now=0.1) is not None  # 1.2.3.4 already used its one slot

    def test_allows_a_request_again_once_the_window_has_elapsed(self):
        from api.main import _UploadRateLimiter

        limiter = _UploadRateLimiter(max_requests=1, window_seconds=60, max_tracked_clients=100)
        limiter.check("1.2.3.4", now=0.0)

        assert limiter.check("1.2.3.4", now=59.0) is not None  # still within the window
        assert limiter.check("1.2.3.4", now=60.1) is None  # window has elapsed

    def test_evicts_the_oldest_tracked_client_once_over_capacity(self):
        from api.main import _UploadRateLimiter

        limiter = _UploadRateLimiter(max_requests=5, window_seconds=60, max_tracked_clients=2)
        limiter.check("first", now=0.0)
        limiter.check("second", now=0.0)
        limiter.check("third", now=0.0)  # should evict "first"

        assert limiter.check("first", now=0.1) is None  # "first" was evicted, so this reads as a brand-new client
        assert len(limiter._requests_by_client) == 2

    def test_does_not_evict_a_client_that_was_touched_recently(self):
        """Regression coverage for a security-review finding: pure
        FIFO-by-insertion-order eviction let an attacker churn through
        >max_tracked_clients distinct keys to force-evict (and thereby
        silently reset the rate limit of) a legitimate client that was
        still genuinely active. Touching a client must refresh its
        recency, so only truly-idle entries are ever evicted.
        """
        from api.main import _UploadRateLimiter

        limiter = _UploadRateLimiter(max_requests=5, window_seconds=60, max_tracked_clients=2)
        limiter.check("first", now=0.0)
        limiter.check("second", now=0.0)
        limiter.check("first", now=0.1)  # re-touching "first" refreshes its recency
        limiter.check("third", now=0.2)  # over capacity now -- must evict "second", not "first"

        assert "first" in limiter._requests_by_client
        assert "second" not in limiter._requests_by_client
        assert len(limiter._requests_by_client) == 2


class TestUploadRateLimitIntegration:
    """A couple of tests through the real HTTP endpoint to confirm the
    wiring (status code, Retry-After header, error body) -- the counting
    logic itself is covered exhaustively above against the plain class.
    """

    def test_returns_429_with_retry_after_once_the_limit_is_exceeded(self, tmp_path: Path, model_dir: Path):
        app = create_app(model_dir=model_dir, records_dir=tmp_path / "records", rate_limit_max_requests=2)
        with TestClient(app) as client:
            _upload_record_230(client)
            _upload_record_230(client)

            response = client.post(
                "/records",
                files={
                    "hea_file": ("230.hea", _read_bytes("Data/Dataset/Train/230.hea")),
                    "dat_file": ("230.dat", _read_bytes("Data/Dataset/Train/230.dat")),
                },
            )

            assert response.status_code == 429
            assert "Retry-After" in response.headers
            assert response.json()["detail"]

    def test_does_not_rate_limit_get_endpoints(self, tmp_path: Path, model_dir: Path):
        app = create_app(model_dir=model_dir, records_dir=tmp_path / "records", rate_limit_max_requests=1)
        with TestClient(app) as client:
            record_id = _upload_record_230(client)  # uses up the only POST slot

            blocked = client.post(
                "/records",
                files={
                    "hea_file": ("230.hea", _read_bytes("Data/Dataset/Train/230.hea")),
                    "dat_file": ("230.dat", _read_bytes("Data/Dataset/Train/230.dat")),
                },
            )
            assert blocked.status_code == 429  # confirms the limit really is exhausted

            # GETs against the same app must still work freely even so.
            for _ in range(5):
                assert client.get(f"/records/{record_id}/beats").status_code == 200
                assert client.get(f"/records/{record_id}/signal").status_code == 200
                assert client.get("/health").status_code == 200
