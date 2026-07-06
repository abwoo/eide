from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class MetricPoint(BaseModel):
    step: int = 0
    value: float = 0.0
    timestamp: datetime | None = None


class EnvironmentInfo(BaseModel):
    os: str = ""
    python_version: str = ""
    runtime_versions: dict[str, str] = {}
    hardware: dict[str, str] = {}
    packages: dict[str, str] = {}


class PipelineStep(BaseModel):
    name: str
    type: str = ""
    parameters: dict[str, Any] = {}
    inputs: list[str] = []
    outputs: list[str] = []
    duration_seconds: float | None = None


class PipelineSpec(BaseModel):
    name: str = ""
    steps: list[PipelineStep] = []
    total_duration_seconds: float | None = None


class TraceEvent(BaseModel):
    step: str
    timestamp: datetime | None = None
    action: str = ""
    details: dict[str, Any] = {}


class ExperimentOutputs(BaseModel):
    metrics: dict[str, float] = {}
    metric_history: dict[str, list[MetricPoint]] = {}
    figures: list[str] = []
    artifacts: list[str] = []
    summary: str = ""
    files: dict[str, str] = {}
