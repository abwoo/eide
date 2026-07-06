from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eide.core.types import PipelineSpec, PipelineStep


class PipelineExecutor:
    def __init__(self, work_dir: str | Path, python_exe: str | None = None):
        self.work_dir = Path(work_dir)
        self.python_exe = python_exe or sys.executable

    def execute(self, pipeline: PipelineSpec) -> list[dict[str, Any]]:
        """Execute all pipeline steps and return execution results."""
        results = []
        for step in pipeline.steps:
            result = self._execute_step(step)
            results.append(result)
        return results

    def _execute_step(self, step: PipelineStep) -> dict[str, Any]:
        start = time.time()
        step_type = step.type.lower() if step.type else ""

        try:
            if step_type == "shell" or step_type == "cmd":
                output, returncode = self._run_shell(step)
            elif step_type == "python" or step_type == "py":
                output, returncode = self._run_python(step)
            elif step_type in ("ml", "training", "eval", "data", "preprocess"):
                output, returncode = self._run_script(step)
            else:
                output, returncode = self._run_script(step)
        except Exception as e:
            elapsed = time.time() - start
            return {
                "step_name": step.name,
                "success": False,
                "duration_seconds": round(elapsed, 2),
                "error": str(e),
            }

        elapsed = time.time() - start
        return {
            "step_name": step.name,
            "success": returncode == 0,
            "duration_seconds": round(elapsed, 2),
            "returncode": returncode,
            "output": output[:2000] if output else "",
        }

    def _run_shell(self, step: PipelineStep) -> tuple[str, int]:
        cmd = step.name
        result = subprocess.run(
            cmd, shell=True, cwd=self.work_dir,
            capture_output=True, text=True, timeout=3600,
        )
        output = result.stdout + result.stderr
        return output, result.returncode

    def _run_python(self, step: PipelineStep) -> tuple[str, int]:
        code = step.parameters.get("code", "")
        script_path = step.parameters.get("script", "")
        if code:
            script_file = self.work_dir / f"_eide_step_{step.name}.py"
            script_file.write_text(code, encoding="utf-8")
            target = script_file
        elif script_path:
            target = Path(script_path)
            if not target.is_absolute():
                target = self.work_dir / script_path
        else:
            return "(no Python code or script specified)", 1
        result = subprocess.run(
            [self.python_exe, str(target)],
            cwd=self.work_dir, capture_output=True, text=True, timeout=3600,
        )
        output = result.stdout + result.stderr
        if code:
            target.unlink(missing_ok=True)
        return output, result.returncode

    def _run_script(self, step: PipelineStep) -> tuple[str, int]:
        script_path = step.parameters.get("script", step.name)
        if not script_path:
            return "(no script specified)", 1

        script_full = Path(script_path)
        if not script_full.is_absolute():
            script_full = self.work_dir / script_path

        if not script_full.exists():
            return f"Script not found: {script_full}", 1

        suffix = script_full.suffix.lower()
        if suffix in (".py",):
            result = subprocess.run(
                [self.python_exe, str(script_full)],
                cwd=self.work_dir, capture_output=True, text=True, timeout=3600,
            )
        else:
            result = subprocess.run(
                [str(script_full)], shell=True,
                cwd=self.work_dir, capture_output=True, text=True, timeout=3600,
            )

        output = result.stdout + result.stderr
        return output, result.returncode
