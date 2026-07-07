from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class ChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class DiffCategory(str, Enum):
    PARAMETER = "parameter"
    STRUCTURAL = "structural"
    ENVIRONMENT = "environment"
    OUTPUT = "output"
    DATA_VERSION = "data_version"
    SEMANTIC = "semantic"


class DiffChange(BaseModel):
    category: DiffCategory
    change_type: ChangeType
    key: str
    old_value: Any = None
    new_value: Any = None
    significance: float = 0.5
    description: str = ""

    @property
    def delta(self) -> float | None:
        if isinstance(self.old_value, (int, float)) and isinstance(self.new_value, (int, float)):
            return self.new_value - self.old_value
        return None


class ImpactEstimate(BaseModel):
    metric: str = ""
    change_direction: str = ""
    change_percent: float = 0.0
    likely_cause: str = ""
    confidence: float = 0.0


class DiffReport(BaseModel):
    experiment_a: str = ""
    experiment_b: str = ""
    changes: list[DiffChange] = []
    summary: str = ""
    impact_estimates: list[ImpactEstimate] = []
    changelog: list[str] = []
