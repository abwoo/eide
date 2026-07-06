from __future__ import annotations

import functools
import inspect
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from eide.capture.manual import _detect_environment
from eide.core.ir import ExperimentIR
from eide.core.types import MetricPoint, PipelineSpec, PipelineStep, TraceEvent
from eide.storage.filestore import FileStore


class ExperimentLogger:
    def __init__(self, experiment: ExperimentIR):
        self.experiment = experiment
        self._start_time = time.time()
        self._step = 0

    def log_metric(self, key: str, value: float):
        self.experiment.outputs.metrics[key] = value
        self._log_metric_history(key, value)

    def log_metrics(self, metrics: dict[str, float]):
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_metric_step(self, key: str, value: float, step: int | None = None):
        """Log a metric at a specific step (for training loops)."""
        s = step if step is not None else self._step
        self.experiment.outputs.metrics[key] = value
        history = self.experiment.outputs.metric_history.setdefault(key, [])
        history.append(MetricPoint(step=s, value=value, timestamp=datetime.now(timezone.utc)))
        self._step = max(self._step, s + 1) if step is None else self._step

    def _log_metric_history(self, key: str, value: float):
        history = self.experiment.outputs.metric_history.setdefault(key, [])
        history.append(MetricPoint(step=self._step, value=value, timestamp=datetime.now(timezone.utc)))
        self._step += 1

    def log_param(self, key: str, value: Any):
        self.experiment.parameters[key] = value

    def log_params(self, params: dict[str, Any]):
        self.experiment.parameters.update(params)

    def log_figure(self, path: str | Path):
        self.experiment.outputs.figures.append(str(path))

    def log_artifact(self, path: str | Path):
        self.experiment.outputs.artifacts.append(str(path))

    def log_tag(self, key: str, value: str):
        self.experiment.tags[key] = value

    def log_decision(self, key: str, value: Any):
        self.experiment.decisions[key] = value

    def elapsed(self) -> float:
        return time.time() - self._start_time


def track(
    name: str | None = None,
    description: str = "",
    store_root: str | None = None,
    auto_save: bool = True,
    capture_args: bool = True,
    capture_return: bool = True,
    capture_env: bool = True,
    data_versions: dict[str, str] | None = None,
):
    """Decorator that auto-captures experiments from function execution.

    Usage:
        @track(name="my-training")
        def train(lr=0.01, epochs=10):
            ...
            return {"accuracy": 0.92}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            exp_name = name or func.__name__

            exp = ExperimentIR(
                name=exp_name,
                description=description,
                timestamp=datetime.now(timezone.utc),
                data_versions=data_versions or {},
            )

            if capture_env:
                exp.environment = _detect_environment()

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if capture_args:
                for param_name, param_value in bound.arguments.items():
                    if isinstance(param_value, (str, int, float, bool, list, dict)):
                        exp.parameters[param_name] = param_value

            exp.pipeline = PipelineSpec(
                name=exp_name,
                steps=[PipelineStep(name=func.__name__, type="python")],
            )

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
            except Exception as e:
                elapsed = time.time() - start_time
                exp.decisions["error"] = str(e)
                exp.decisions["success"] = False
                exp.metadata["duration_seconds"] = round(elapsed, 2)
                exp.metadata["success"] = False
                exp.execution_trace.append(
                    TraceEvent(step=func.__name__, action="error", details={"error": str(e), "duration": round(elapsed, 2)})
                )
                if auto_save:
                    _save_exp(exp, store_root)
                raise

            exp.decisions["success"] = True
            exp.metadata["duration_seconds"] = round(elapsed, 2)
            exp.metadata["success"] = True
            exp.execution_trace.append(
                TraceEvent(step=func.__name__, action="complete", details={"duration": round(elapsed, 2)})
            )

            if capture_return and result is not None:
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, (int, float)):
                            exp.outputs.metrics[k] = v
                        elif isinstance(v, str):
                            exp.outputs.files[k] = v
                elif isinstance(result, (int, float)):
                    exp.outputs.metrics["result"] = result

            if auto_save:
                _save_exp(exp, store_root)

            return exp, result

        return wrapper

    if callable(name):
        func = name
        name = None
        return decorator(func)

    return decorator


def _save_exp(exp: ExperimentIR, store_root: str | None = None):
    root = Path(store_root or os.environ.get("EIDE_ROOT", ".eide"))
    store = FileStore(root)
    store.save(exp)
