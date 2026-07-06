from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

from eide.capture.base import CaptureAdapter
from eide.core.ir import ExperimentIR
from eide.core.types import EnvironmentInfo


def _detect_hardware() -> dict[str, str]:
    hw: dict[str, str] = {
        "cpu": platform.processor() or "",
        "cpu_count": str(os.cpu_count() or 0),
        "machine": platform.machine(),
    }

    try:
        import psutil
        hw["ram_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
        hw["ram_available_gb"] = f"{psutil.virtual_memory().available / (1024**3):.1f}"
    except ImportError:
        pass

    try:
        import torch
        hw["torch_version"] = torch.__version__
        hw["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            hw["cuda_version"] = torch.version.cuda or ""
            hw["gpu_count"] = str(torch.cuda.device_count())
            hw["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else ""
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for i, line in enumerate(lines):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    hw[f"gpu_{i}"] = parts[0]
                    if len(parts) > 1:
                        hw[f"gpu_{i}_memory"] = parts[1]
                    if len(parts) > 2 and i == 0:
                        hw["nvidia_driver"] = parts[2]
    except Exception:
        pass

    return hw


def _detect_git() -> dict[str, str]:
    git_info: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()[:12]

        result2 = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True, timeout=3,
        )
        if result2.returncode == 0 and result2.stdout.strip():
            git_info["dirty"] = "true"

        result3 = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=3,
        )
        if result3.returncode == 0 and result3.stdout.strip():
            git_info["branch"] = result3.stdout.strip()
    except Exception:
        pass
    return git_info


def _detect_conda() -> dict[str, str]:
    conda_info: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["conda", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            conda_info["conda_version"] = result.stdout.strip()
            env_result = subprocess.run(
                ["conda", "info", "--envs", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if env_result.returncode == 0:
                import json
                data = json.loads(env_result.stdout)
                conda_info["active_env"] = data.get("active_prefix_name", "")
                conda_info["env_count"] = str(len(data.get("envs", [])))
    except Exception:
        pass
    return conda_info


def _detect_environment() -> EnvironmentInfo:
    hw = _detect_hardware()

    runtime_versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    git_info = _detect_git()
    runtime_versions.update(git_info)

    conda_info = _detect_conda()
    runtime_versions.update(conda_info)

    info = EnvironmentInfo(
        os=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
        runtime_versions=runtime_versions,
        hardware=hw,
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            import json
            packages = json.loads(result.stdout)
            info.packages = {pkg["name"]: pkg["version"] for pkg in packages}
    except Exception:
        pass

    info.packages["python"] = sys.version.split()[0]

    return info


class ManualCapture(CaptureAdapter):
    def capture(
        self,
        name: str = "",
        description: str = "",
        project: str = "",
        parameters: dict[str, Any] | None = None,
        pipeline_steps: list[dict[str, Any]] | None = None,
        metrics: dict[str, float] | None = None,
        figures: list[str] | None = None,
        tags: dict[str, str] | None = None,
        data_versions: dict[str, str] | None = None,
        decisions: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentIR:
        env = _detect_environment()
        exp = ExperimentIR(
            name=name,
            description=description,
            project=project,
            environment=env,
            parameters=parameters or {},
            data_versions=data_versions or {},
            decisions=decisions or {},
            tags=tags or {},
            metadata=metadata or {},
        )
        if pipeline_steps:
            from eide.core.types import PipelineSpec, PipelineStep

            exp.pipeline = PipelineSpec(
                name=name,
                steps=[PipelineStep(**s) for s in pipeline_steps],
            )
        if metrics:
            exp.outputs.metrics = metrics
        if figures:
            exp.outputs.figures = figures
        return exp


def capture_manual(
    name: str = "",
    description: str = "",
    project: str = "",
    parameters: dict[str, Any] | None = None,
    pipeline_steps: list[dict[str, Any]] | None = None,
    metrics: dict[str, float] | None = None,
    figures: list[str] | None = None,
    tags: dict[str, str] | None = None,
    data_versions: dict[str, str] | None = None,
    decisions: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExperimentIR:
    """Quick one-shot manual capture"""
    return ManualCapture().capture(
        name=name,
        description=description,
        project=project,
        parameters=parameters,
        pipeline_steps=pipeline_steps,
        metrics=metrics,
        figures=figures,
        tags=tags,
        data_versions=data_versions,
        decisions=decisions,
        metadata=metadata,
    )
