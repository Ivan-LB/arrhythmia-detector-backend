"""FastAPI application: record upload, beat classification, signal retrieval.

`create_app()` is a factory, not a fixed module-level instance, so tests
can point it at a small/fast model directory instead of a real one. The
module-level `app` (for `uvicorn api.main:app`) is constructed lazily via
`__getattr__` (PEP 562) so merely importing this module -- e.g. `from
api.main import create_app` in a test -- never requires MODEL_DIR to be
set. Actually running the service does require it, deliberately: there is
no reasonable degraded mode for an ECG classifier API with no model, so
failing clearly at startup is correct here, unlike the original project's
bug of crashing at bare import time with a hardcoded, wrong path and no
error message at all.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import wfdb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from api.inference import InferenceBundle, diagnose_beats, load_inference_bundle
from api.records import MAX_UPLOAD_SIZE_BYTES, InvalidRecordUpload, StoredRecord, save_uploaded_record
from api.schemas import BeatPrediction, BeatsResponse, HealthResponse, RecordMetadata, SignalResponse
from ecg_pipeline import preprocessing
from ecg_pipeline.labels import AAMI_CLASSES

logger = logging.getLogger(__name__)

SIGNAL_DOWNSAMPLE_TARGET_POINTS = 2000
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024  # 1MB
DEFAULT_MAX_STORED_RECORDS = 50
# Local dev default only -- matches arrhythmia-detector-web's dev server.
# Production deployments must override via the CORS_ALLOWED_ORIGINS env var;
# there is no wildcard fallback, deliberately (see create_app's docstring).
DEFAULT_CORS_ALLOWED_ORIGINS: tuple[str, ...] = ("http://localhost:3000",)


@dataclass
class _RecordEntry:
    stored: StoredRecord
    metadata: RecordMetadata
    cached_beats: BeatsResponse | None = None
    cached_signal: SignalResponse | None = None


async def _read_upload_within_limit(
    upload: UploadFile, label: str, max_bytes: int = MAX_UPLOAD_SIZE_BYTES
) -> bytes:
    """Read an UploadFile in bounded chunks, aborting as soon as the total
    exceeds max_bytes, rather than reading the whole body into memory first
    and only checking size afterward.

    An unbounded `await upload.read()` lets a single request's body --
    which Starlette will spool to disk past 1MB with no total-size ceiling
    of its own -- be fully materialized in process memory before any
    app-level size check ever runs. That's a memory-exhaustion DoS
    triggerable by a single unauthenticated request with no special
    tooling, confirmed via direct testing of the underlying multipart
    parser. Reading in bounded chunks with an early abort keeps peak
    memory bounded to roughly max_bytes + one chunk, regardless of how
    large the client claims (or attempts) to send.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(UPLOAD_CHUNK_SIZE_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise InvalidRecordUpload(f"{label} exceeds the maximum upload size of {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _require_filename(upload: UploadFile) -> str:
    """Starlette rejects any multipart file part with a missing/empty
    filename with its own 422 before the endpoint body ever runs (verified
    directly), so this is unreachable in practice -- but UploadFile.filename
    is typed str | None, and asserting it here (rather than indexing into
    an Optional without narrowing) keeps that guarantee explicit and
    type-checked instead of implicit and undocumented.
    """
    if upload.filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
    return upload.filename


def create_app(
    model_dir: Path,
    records_dir: Path,
    max_stored_records: int = DEFAULT_MAX_STORED_RECORDS,
    cors_allowed_origins: tuple[str, ...] = DEFAULT_CORS_ALLOWED_ORIGINS,
) -> FastAPI:
    """Build the FastAPI app against a specific model directory and a
    directory to store uploaded records under.

    cors_allowed_origins is an explicit allowlist, never a wildcard: this
    API sends no cookies/session credentials, so allow_credentials stays
    False, but reflecting an unbounded '*' origin back is still needless
    exposure for an endpoint that accepts file uploads.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = load_inference_bundle(model_dir)
        app.state.records = {}  # dict[str, _RecordEntry] -- can't annotate a non-self attribute inline
        yield

    app = FastAPI(title="Arrhythmia Detector Backend", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def _get_record_entry(record_id: str) -> _RecordEntry:
        entry = app.state.records.get(record_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Record {record_id!r} not found")
        return entry

    def _evict_oldest_if_over_capacity() -> None:
        """Bound both memory (app.state.records) and disk (records_dir)
        growth: with no eviction, every accepted upload lives for the
        entire process lifetime with nothing ever removing it -- a
        long-running deployment accumulates both without limit. FIFO
        eviction (Python dicts preserve insertion order) is a simple,
        proportionate fix for a portfolio/demo-scale service; a full
        TTL/LRU policy would be over-engineering for the current scope.
        """
        records = app.state.records
        while len(records) > max_stored_records:
            oldest_id, oldest_entry = next(iter(records.items()))
            del records[oldest_id]
            shutil.rmtree(oldest_entry.stored.record_path.parent, ignore_errors=True)
            logger.info("Evicted record %s (capacity %d reached)", oldest_id, max_stored_records)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        bundle: InferenceBundle = app.state.bundle
        return HealthResponse(status="ok", model_version=bundle.model_version, aami_classes=list(AAMI_CLASSES))

    @app.post("/records", response_model=RecordMetadata)
    async def upload_record(
        hea_file: UploadFile = File(...),
        dat_file: UploadFile = File(...),
        atr_file: UploadFile | None = File(None),
    ) -> RecordMetadata:
        try:
            hea_content = await _read_upload_within_limit(hea_file, "hea file")
            dat_content = await _read_upload_within_limit(dat_file, "dat file")
            atr_content = await _read_upload_within_limit(atr_file, "atr file") if atr_file is not None else None

            stored = await run_in_threadpool(
                save_uploaded_record,
                records_dir,
                _require_filename(hea_file),
                hea_content,
                _require_filename(dat_file),
                dat_content,
                atr_filename=_require_filename(atr_file) if atr_file is not None else None,
                atr_content=atr_content,
            )
        except InvalidRecordUpload as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        # save_uploaded_record already confirmed this parses (and blocking
        # I/O + wfdb's C-accelerated parsing shouldn't run on the event
        # loop thread), so offload this read the same way.
        record = await run_in_threadpool(wfdb.rdrecord, str(stored.record_path))
        metadata = RecordMetadata(
            record_id=stored.record_id,
            duration_seconds=record.sig_len / record.fs,
            sampling_rate=float(record.fs),
            lead_names=record.sig_name,
        )
        app.state.records[stored.record_id] = _RecordEntry(stored=stored, metadata=metadata)
        _evict_oldest_if_over_capacity()
        return metadata

    @app.get("/records/{record_id}/beats", response_model=BeatsResponse)
    def get_beats(record_id: str) -> BeatsResponse:
        entry = _get_record_entry(record_id)
        if entry.cached_beats is not None:
            return entry.cached_beats

        try:
            record = wfdb.rdrecord(str(entry.stored.record_path))
            mlii_index = preprocessing.find_mlii_channel(record.sig_name)
            signal = record.p_signal[:, mlii_index]

            beat_source: Literal["annotations", "detected"]
            if entry.stored.has_annotations:
                annotation = wfdb.rdann(str(entry.stored.record_path), "atr")
                beat_samples = annotation.sample
                beat_source = "annotations"
            else:
                filtered = preprocessing.apply_notch_filter(preprocessing.remove_baseline(signal), record.fs)
                beat_samples = preprocessing.detect_r_peaks(filtered, record.fs)
                beat_source = "detected"
        except Exception as error:
            logger.exception("Failed to read/process record %s", record_id)
            raise HTTPException(status_code=500, detail=f"Failed to process record {record_id!r}") from error

        bundle: InferenceBundle = app.state.bundle
        results = diagnose_beats(bundle, signal, record.fs, beat_samples)

        beats = [
            BeatPrediction(
                sample_index=result.sample_index,
                time_seconds=result.sample_index / record.fs,
                aami_class=result.aami_class,
                confidence=result.confidence,
            )
            for result in results
        ]
        response = BeatsResponse(record_id=record_id, beat_source=beat_source, beats=beats)
        entry.cached_beats = response
        return response

    @app.get("/records/{record_id}/signal", response_model=SignalResponse)
    def get_signal(record_id: str) -> SignalResponse:
        entry = _get_record_entry(record_id)
        if entry.cached_signal is not None:
            return entry.cached_signal

        try:
            record = wfdb.rdrecord(str(entry.stored.record_path))
            mlii_index = preprocessing.find_mlii_channel(record.sig_name)
            signal = record.p_signal[:, mlii_index]
            filtered = preprocessing.apply_notch_filter(preprocessing.remove_baseline(signal), record.fs)
        except Exception as error:
            logger.exception("Failed to read/process record %s", record_id)
            raise HTTPException(status_code=500, detail=f"Failed to process record {record_id!r}") from error

        downsample_factor = max(1, len(filtered) // SIGNAL_DOWNSAMPLE_TARGET_POINTS)
        downsampled = filtered[::downsample_factor]

        response = SignalResponse(
            record_id=record_id,
            sampling_rate=record.fs,
            downsample_factor=downsample_factor,
            samples=downsampled.tolist(),
        )
        entry.cached_signal = response
        return response

    return app


def _default_records_dir() -> Path:
    records_dir = Path(tempfile.gettempdir()) / "arrhythmia-detector-backend" / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    return records_dir


def _parse_cors_allowed_origins(cors_env: str | None) -> tuple[str, ...]:
    """Parse the CORS_ALLOWED_ORIGINS env var (comma-separated) into an
    origin tuple, falling back to DEFAULT_CORS_ALLOWED_ORIGINS when unset
    or when every comma-segment is blank after stripping (e.g. "," or " ")
    -- prevents a typo'd env var from silently locking out every browser
    origin via allow_origins=[].
    """
    if not cors_env:
        return DEFAULT_CORS_ALLOWED_ORIGINS
    parsed = tuple(origin.strip() for origin in cors_env.split(",") if origin.strip())
    if not parsed:
        logger.warning(
            "CORS_ALLOWED_ORIGINS was set but contained no usable origins after parsing "
            "(%r); falling back to the default %r.",
            cors_env,
            DEFAULT_CORS_ALLOWED_ORIGINS,
        )
        return DEFAULT_CORS_ALLOWED_ORIGINS
    return parsed


def _app_from_env() -> FastAPI:
    import os

    model_dir_env = os.environ.get("MODEL_DIR")
    if not model_dir_env:
        raise RuntimeError(
            "MODEL_DIR environment variable must be set to a specific, versioned model "
            "directory (e.g. models/beat-classifier-20260705-dc9f983). There is no "
            "implicit default -- silently picking 'whatever is in models/' is exactly "
            "the untraceable-artifact problem this project's model versioning scheme "
            "exists to fix."
        )

    return create_app(
        model_dir=Path(model_dir_env),
        records_dir=_default_records_dir(),
        cors_allowed_origins=_parse_cors_allowed_origins(os.environ.get("CORS_ALLOWED_ORIGINS")),
    )


def __getattr__(name: str):
    if name == "app":
        return _app_from_env()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
