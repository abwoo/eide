import tempfile
from pathlib import Path

from eide.core.ir import ExperimentIR
from eide.core.types import (
    ExperimentOutputs,
    PipelineSpec,
    PipelineStep,
)
from eide.replay.engine import ReplayEngine
from eide.replay.environment import EnvironmentManager
from eide.replay.executor import PipelineExecutor
from eide.replay.verifier import ReplayVerifier
from eide.storage.filestore import FileStore


class TestEnvironmentManager:
    def test_detect_current(self):
        mgr = EnvironmentManager()
        info = mgr.detect_current()
        assert info.python_version
        assert len(info.packages) > 0

    def test_python_exe_none_before_restore(self):
        mgr = EnvironmentManager()
        assert mgr.python_exe() is None


class TestPipelineExecutor:
    def test_execute_simple_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = PipelineExecutor(tmp)
            script = Path(tmp) / "test_script.py"
            script.write_text("print('hello from replay')", encoding="utf-8")

            step = PipelineStep(
                name=str(script),
                type="python",
                parameters={"script": str(script)},
            )
            result = executor._execute_step(step)
            assert result["success"] is True
            assert "hello from replay" in result.get("output", "")

    def test_execute_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "step1.py"
            script.write_text("print('step1 ok')", encoding="utf-8")

            pipeline = PipelineSpec(
                steps=[PipelineStep(name=str(script), type="python", parameters={"script": str(script)})],
            )
            executor = PipelineExecutor(tmp)
            results = executor.execute(pipeline)
            assert len(results) == 1
            assert results[0]["success"] is True


class TestReplayVerifier:
    def test_identical_experiments(self):
        exp = ExperimentIR(
            name="test",
            parameters={"lr": 0.01},
            outputs=ExperimentOutputs(metrics={"acc": 0.9}),
        )
        verifier = ReplayVerifier()
        verdict = verifier.verify(exp, exp)
        assert verdict.is_identical is True
        assert "identical" in verdict.summary

    def test_diverged_experiments(self):
        exp_a = ExperimentIR(
            name="original",
            parameters={"lr": 0.01},
            outputs=ExperimentOutputs(metrics={"acc": 0.9}),
        )
        exp_b = ExperimentIR(
            name="replay",
            parameters={"lr": 0.01},
            outputs=ExperimentOutputs(metrics={"acc": 0.85}),
        )
        verifier = ReplayVerifier()
        verdict = verifier.verify(exp_a, exp_b)
        assert verdict.is_identical is False
        assert verdict.output_changes > 0
        assert "diverged" in verdict.summary


class TestReplayEngine:
    def test_replay_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp) / "store"
            store = FileStore(store_dir)

            script = Path(tmp) / "train.py"
            script.write_text("print('training complete')", encoding="utf-8")

            original = ExperimentIR(
                name="test-run",
                parameters={"lr": 0.01, "epochs": 5},
                pipeline=PipelineSpec(
                    name="test-pipe",
                    steps=[PipelineStep(name=str(script), type="python", parameters={"script": str(script)})],
                ),
                outputs=ExperimentOutputs(metrics={"accuracy": 0.9}),
                tags={"project": "eide-test"},
            )
            store.save(original)

            engine = ReplayEngine(
                store=store,
                work_dir=Path(tmp) / "replay_work",
            )
            result = engine.replay(original, restore_environment=False, restore_data=False)

            assert result.replay_experiment is not None
            assert result.verdict is not None
            assert len(result.step_results) > 0
            assert "replay_of" in result.replay_experiment.tags
            assert result.replay_experiment.tags["replay_of"] == original.id
