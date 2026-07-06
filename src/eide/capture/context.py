from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from eide.capture.decorator import ExperimentLogger
from eide.capture.manual import _detect_environment
from eide.capture.resource_monitor import ResourceMonitor
from eide.core.ir import ExperimentIR
from eide.storage.filestore import FileStore


class ExperimentContext:
    """Context manager for experiment capture.

    Usage:
        with ExperimentContext("my-run") as exp:
            exp.log_param("lr", 0.01)
            exp.log_metric("accuracy", 0.92)
            # auto-saves on exit
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        project: str = "",
        store_root: str | None = None,
        auto_save: bool = True,
        capture_env: bool = True,
        monitor_resources: bool = False,
        resource_interval: float = 5.0,
    ):
        self.name = name
        self.description = description
        self.project = project
        self.store_root = Path(store_root or os.environ.get("EIDE_ROOT", ".eide"))
        self.auto_save = auto_save
        self.capture_env = capture_env
        self.monitor_resources = monitor_resources
        self.resource_interval = resource_interval
        self.experiment: ExperimentIR | None = None
        self.logger: ExperimentLogger | None = None
        self._start_time: float | None = None
        self._monitor: ResourceMonitor | None = None

    def __enter__(self) -> ExperimentLogger:
        self._start_time = time.time()
        self.experiment = ExperimentIR(
            name=self.name,
            description=self.description,
            project=self.project,
            timestamp=datetime.now(timezone.utc),
        )
        if self.capture_env:
            self.experiment.environment = _detect_environment()
        self.logger = ExperimentLogger(self.experiment)
        if self.monitor_resources:
            self._monitor = ResourceMonitor(interval=self.resource_interval)
            self._monitor.start()
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start_time if self._start_time else 0
        self.experiment.metadata["duration_seconds"] = round(elapsed, 2)
        self.experiment.metadata["success"] = exc_type is None
        if exc_type is not None:
            self.experiment.decisions["error"] = str(exc_val)

        if self._monitor:
            snapshots = self._monitor.stop()
            self.experiment.metadata["resource_snapshots"] = snapshots
            if snapshots:
                cpu_vals = [s.get("cpu_percent", 0) for s in snapshots if "cpu_percent" in s]
                ram_vals = [s.get("ram_percent", 0) for s in snapshots if "ram_percent" in s]
                if cpu_vals:
                    self.experiment.metadata["avg_cpu_percent"] = round(sum(cpu_vals) / len(cpu_vals), 1)
                    self.experiment.metadata["max_cpu_percent"] = round(max(cpu_vals), 1)
                if ram_vals:
                    self.experiment.metadata["avg_ram_percent"] = round(sum(ram_vals) / len(ram_vals), 1)

        if self.auto_save:
            store = FileStore(self.store_root)
            store.save(self.experiment)
