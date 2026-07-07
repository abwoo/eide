from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any


class ResourceMonitor:
    """Monitor CPU, RAM, and GPU usage over time in a background thread.

    Records MetricPoint-like snapshots stored in experiment metadata.
    """

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.snapshots: list[dict[str, Any]] = []

    def _sample(self) -> dict[str, Any]:
        snap: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            import psutil

            snap["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            snap["ram_percent"] = mem.percent
            snap["ram_used_gb"] = round(mem.used / (1024**3), 2)
            snap["ram_total_gb"] = round(mem.total / (1024**3), 2)
        except ImportError:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    snap[f"gpu_{i}_util_percent"] = torch.cuda.utilization(i)
                    snap[f"gpu_{i}_mem_used_gb"] = round(
                        torch.cuda.memory_allocated(i) / (1024**3), 2
                    )
                    snap[f"gpu_{i}_mem_total_gb"] = round(
                        torch.cuda.get_device_properties(i).total_memory / (1024**3), 2
                    )
        except ImportError:
            pass
        return snap

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        with self._lock:
            self.snapshots = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=10)
            self._thread = None
        with self._lock:
            return list(self.snapshots)

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                self.snapshots.append(self._sample())
            self._stop.wait(self.interval)
