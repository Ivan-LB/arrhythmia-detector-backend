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
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import wfdb
from fastapi import FastAPI, File, HTTPException, UploadFile

from api.inference import InferenceBundle, diagnose_beats, load_inference_bundle
from api.records import InvalidRecordUpload, StoredRecord, save_uploaded_record
from api.schemas import BeatPrediction, BeatsResponse, HealthResponse, RecordMetadata, SignalResponse
from ecg_pipeline import preprocessing
from ecg_pipeline.labels import AAMI_CLASSES

logger = logging.getLogger(__name__)

SIGNAL_DOWNSAMPLE_TARGET_POINTS = 2000


@dataclass(frozen=True)
class _RecordEntry:
    stored: StoredRecord
    metadata: RecordMetadata


def create_app(model_dir: Path, records_dir: Path) -> FastAPI:
    """Build the FastAPI app against a specific model directory and a
    directory to store uploaded records under.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = load_inference_bundle(model_dir)
        app.state.records = {}
        yield

    app = FastAPI(title="Arrhythmia Detector Backend", lifespan=lifespan)

    def _get_record_entry(record_id: str) -> _RecordEntry:
        entry = app.state.records.get(record_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Record {record_id!r} not found")
        return entry

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
        hea_content = await hea_file.read()
        dat_content = await dat_file.read()
        atr_content = await atr_file.read() if atr_file is not None else None

        try:
            stored = save_uploaded_record(
                records_dir,
                hea_file.filename,
                hea_content,
                dat_file.filename,
                dat_content,
                atr_filename=atr_file.filename if atr_file is not None else None,
                atr_content=atr_content,
            )
        except InvalidRecordUpload as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        record = wfdb.rdrecord(str(stored.record_path))
        metadata = RecordMetadata(
            record_id=stored.record_id,
            duration_seconds=record.sig_len / record.fs,
            sampling_rate=float(record.fs),
            lead_names=record.sig_name,
        )
        app.state.records[stored.record_id] = _RecordEntry(stored=stored, metadata=metadata)
        return metadata

    @app.get("/records/{record_id}/beats", response_model=BeatsResponse)
    def get_beats(record_id: str) -> BeatsResponse:
        entry = _get_record_entry(record_id)
        record = wfdb.rdrecord(str(entry.stored.record_path))
        mlii_index = preprocessing.find_mlii_channel(record.sig_name)
        signal = record.p_signal[:, mlii_index]

        if entry.stored.has_annotations:
            annotation = wfdb.rdann(str(entry.stored.record_path), "atr")
            beat_samples = annotation.sample
            beat_source = "annotations"
        else:
            filtered = preprocessing.apply_notch_filter(preprocessing.remove_baseline(signal), record.fs)
            beat_samples = preprocessing.detect_r_peaks(filtered, record.fs)
            beat_source = "detected"

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
        return BeatsResponse(record_id=record_id, beat_source=beat_source, beats=beats)

    @app.get("/records/{record_id}/signal", response_model=SignalResponse)
    def get_signal(record_id: str) -> SignalResponse:
        entry = _get_record_entry(record_id)
        record = wfdb.rdrecord(str(entry.stored.record_path))
        mlii_index = preprocessing.find_mlii_channel(record.sig_name)
        signal = record.p_signal[:, mlii_index]
        filtered = preprocessing.apply_notch_filter(preprocessing.remove_baseline(signal), record.fs)

        downsample_factor = max(1, len(filtered) // SIGNAL_DOWNSAMPLE_TARGET_POINTS)
        downsampled = filtered[::downsample_factor]

        return SignalResponse(
            record_id=record_id,
            sampling_rate=record.fs,
            downsample_factor=downsample_factor,
            samples=downsampled.tolist(),
        )

    return app


def _default_records_dir() -> Path:
    records_dir = Path(tempfile.gettempdir()) / "arrhythmia-detector-backend" / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    return records_dir


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
    return create_app(model_dir=Path(model_dir_env), records_dir=_default_records_dir())


def __getattr__(name: str):
    if name == "app":
        return _app_from_env()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
