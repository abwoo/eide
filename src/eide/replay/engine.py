from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from eide.capture.manual import capture_manual
from eide.core.ir import ExperimentIR
from eide.core.types import PipelineSpec
from eide.replay.environment import EnvironmentManager
from eide.replay.executor import PipelineExecutor
from eide.replay.verifier import ReplayVerdict, ReplayVerifier
from eide.storage.filestore import FileStore


class ReplayEngine:
    def __init__(
        self,
        store: FileStore | None = None,
        artifacts_root: str | Path | None = None,
        work_dir: str | Path | None = None,
    ):
        self.store = store
        self.artifacts_root = Path(artifacts_root) if artifacts_root else None
        self.work_dir = (
            Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="eide_replay_"))
        )
        self.env_manager = EnvironmentManager(self.work_dir)
        self._setup_dirs()

    def _setup_dirs(self):
        (self.work_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.work_dir / "outputs").mkdir(parents=True, exist_ok=True)

    def replay(
        self,
        experiment: ExperimentIR,
        restore_environment: bool = True,
        restore_data: bool = True,
        save_replay: bool = True,
    ) -> ReplayResult:
        """Replay an experiment and compare with the original."""
        env_restored = False
        if restore_environment and experiment.environment.packages:
            print(
                (
                    f"[replay] Restoring environment"
                    f" ({len(experiment.environment.packages)} packages)..."
                )
            )
            self.env_manager.restore(experiment.environment)
            env_restored = True

        if restore_data and experiment.data_versions:
            print("[replay] Restoring data from version references...")
            self._restore_data(experiment.data_versions)

        python_exe = self.env_manager.python_exe() if env_restored else None
        executor = PipelineExecutor(self.work_dir, python_exe=python_exe)

        print(f"[replay] Executing pipeline ({len(experiment.pipeline.steps)} steps)...")
        step_results = executor.execute(experiment.pipeline)

        print("[replay] Capturing replay result...")
        replay_exp = capture_manual(
            name=f"replay-{experiment.name or experiment.id}",
            description=f"Replay of {experiment.id}",
            parameters=experiment.parameters,
            tags={**experiment.tags, "replay_of": experiment.id},
        )
        replay_exp.pipeline = PipelineSpec(
            name=f"replay-{experiment.pipeline.name}",
            steps=experiment.pipeline.steps,
        )

        for sr in step_results:
            from eide.core.types import TraceEvent

            replay_exp.execution_trace.append(
                TraceEvent(
                    step=sr["step_name"],
                    action="replay_execute",
                    details=sr,
                )
            )

        replay_exp.outputs.files = {}
        outputs_dir = self.work_dir / "outputs"
        if outputs_dir.exists():
            for f in outputs_dir.iterdir():
                if f.is_file():
                    replay_exp.outputs.files[f.name] = str(f)

        if self.artifacts_root:
            replay_exp.outputs.figures = [
                str(f)
                for f in self.artifacts_root.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".svg")
            ]

        if save_replay and self.store:
            self.store.save(replay_exp)
            print(f"[replay] Saved replay as experiment {replay_exp.id}")

        print("[replay] Verifying replay against original...")
        verifier = ReplayVerifier(
            artifacts_root=str(self.artifacts_root) if self.artifacts_root else None
        )
        verdict = verifier.verify(experiment, replay_exp)

        return ReplayResult(
            replay_experiment=replay_exp,
            verdict=verdict,
            step_results=step_results,
            work_dir=self.work_dir,
        )

    def _restore_data(self, data_versions: dict[str, str]):
        for key, ref in data_versions.items():
            ref_path = Path(ref)
            if ref_path.exists():
                dest = self.work_dir / "data" / key
                if ref_path.is_dir():
                    shutil.copytree(ref_path, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(ref_path, dest)

    def cleanup(self):
        self.env_manager.cleanup()
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)


class ReplayResult:
    def __init__(
        self,
        replay_experiment: ExperimentIR,
        verdict: ReplayVerdict,
        step_results: list[dict[str, Any]],
        work_dir: Path,
    ):
        self.replay_experiment = replay_experiment
        self.verdict = verdict
        self.step_results = step_results
        self.work_dir = work_dir

    def to_dict(self) -> dict:
        return {
            "replay_id": self.replay_experiment.id,
            "original_id": self.verdict.original_id,
            "is_identical": self.verdict.is_identical,
            "summary": self.verdict.summary,
            "total_changes": self.verdict.total_changes,
            "step_results": self.step_results,
        }
