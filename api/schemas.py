"""Request/response models for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RecordMetadata(BaseModel):
    record_id: str
    duration_seconds: float
    sampling_rate: float
    lead_names: list[str]


class BeatPrediction(BaseModel):
    sample_index: int
    time_seconds: float
    aami_class: str
    confidence: float = Field(ge=0.0, le=1.0)


class BeatsResponse(BaseModel):
    record_id: str
    beat_source: Literal["annotations", "detected"]
    beats: list[BeatPrediction]


class SignalResponse(BaseModel):
    record_id: str
    sampling_rate: float
    downsample_factor: int
    samples: list[float]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    aami_classes: list[str]
