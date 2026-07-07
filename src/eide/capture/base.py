from __future__ import annotations

from abc import ABC, abstractmethod

from eide.core.ir import ExperimentIR


class CaptureAdapter(ABC):
    @abstractmethod
    def capture(self, **kwargs) -> ExperimentIR: ...
