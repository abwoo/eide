from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from eide.core.types import (
    EnvironmentInfo,
    ExperimentOutputs,
    PipelineSpec,
    TraceEvent,
)


class ExperimentIR(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: str = ""
    description: str = ""
    project: str = ""

    environment: EnvironmentInfo = Field(default_factory=EnvironmentInfo)
    pipeline: PipelineSpec = Field(default_factory=PipelineSpec)
    parameters: dict[str, Any] = {}
    data_versions: dict[str, str] = {}
    decisions: dict[str, Any] = {}
    outputs: ExperimentOutputs = Field(default_factory=ExperimentOutputs)
    execution_trace: list[TraceEvent] = []

    tags: dict[str, str] = {}
    metadata: dict[str, Any] = {}
