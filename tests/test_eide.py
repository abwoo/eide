import tempfile

import pytest

from eide.core.ir import ExperimentIR
from eide.core.types import (
    EnvironmentInfo,
    ExperimentOutputs,
    PipelineSpec,
    PipelineStep,
)
from eide.diff.engine import DiffEngine
from eide.diff.environment import diff_environment
from eide.diff.parameter import diff_parameters
from eide.diff.semantic import diff_metrics
from eide.diff.structural import diff_pipeline
from eide.storage.filestore import FileStore


class TestExperimentIR:
    def test_create_minimal(self):
        exp = ExperimentIR()
        assert exp.id is not None
        assert len(exp.id) == 12
        assert exp.parameters == {}
        assert exp.tags == {}

    def test_create_with_params(self):
        exp = ExperimentIR(
            name="test",
            parameters={"lr": 0.01, "epochs": 10},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.95}),
        )
        assert exp.name == "test"
        assert exp.parameters["lr"] == 0.01
        assert exp.outputs.metrics["accuracy"] == 0.95

    def test_serialize_deserialize(self):
        exp = ExperimentIR(
            name="test",
            parameters={"lr": 0.01},
            tags={"env": "prod"},
        )
        data = exp.model_dump()
        restored = ExperimentIR.model_validate(data)
        assert restored.id == exp.id
        assert restored.parameters == exp.parameters
        assert restored.tags == exp.tags


class TestFileStore:
    @pytest.fixture
    def temp_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield FileStore(tmp)

    def test_save_load(self, temp_store):
        exp = ExperimentIR(name="test-exp", parameters={"lr": 0.01})
        temp_store.save(exp)
        loaded = temp_store.load(exp.id)
        assert loaded.id == exp.id
        assert loaded.parameters == exp.parameters

    def test_list(self, temp_store):
        e1 = ExperimentIR(name="exp-1")
        e2 = ExperimentIR(name="exp-2")
        temp_store.save(e1)
        temp_store.save(e2)
        entries = temp_store.list_experiments()
        assert len(entries) == 2

    def test_delete(self, temp_store):
        exp = ExperimentIR(name="to-delete")
        temp_store.save(exp)
        temp_store.delete(exp.id)
        with pytest.raises(FileNotFoundError):
            temp_store.load(exp.id)


class TestParameterDiff:
    def test_no_changes(self):
        changes = diff_parameters({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert len(changes) == 0

    def test_added_param(self):
        changes = diff_parameters({"a": 1}, {"a": 1, "b": 2})
        assert len(changes) == 1
        assert changes[0].change_type.value == "added"
        assert changes[0].key == "b"

    def test_removed_param(self):
        changes = diff_parameters({"a": 1, "b": 2}, {"a": 1})
        assert len(changes) == 1
        assert changes[0].change_type.value == "removed"

    def test_modified_numeric(self):
        changes = diff_parameters({"a": 1.0}, {"a": 2.0})
        assert len(changes) == 1
        assert changes[0].change_type.value == "modified"
        assert "↑" in changes[0].description or "↓" in changes[0].description

    def test_numeric_tolerance(self):
        changes = diff_parameters({"a": 1.0}, {"a": 1.0001}, tolerances={"a": 1e-3})
        assert len(changes) == 0

    def test_nested_params(self):
        changes = diff_parameters(
            {"optimizer": {"lr": 0.01, "momentum": 0.9}},
            {"optimizer": {"lr": 0.001, "momentum": 0.9}},
        )
        assert len(changes) == 1
        assert "optimizer.lr" in changes[0].key


class TestStructuralDiff:
    def test_identical_pipeline(self):
        steps = [PipelineStep(name="load", type="data"), PipelineStep(name="train", type="ml")]
        pipe = PipelineSpec(steps=steps)
        changes = diff_pipeline(pipe, pipe)
        assert len(changes) == 0

    def test_added_step(self):
        pipe_a = PipelineSpec(steps=[PipelineStep(name="load", type="data")])
        pipe_b = PipelineSpec(steps=[PipelineStep(name="load", type="data"), PipelineStep(name="train", type="ml")])
        changes = diff_pipeline(pipe_a, pipe_b)
        added = [c for c in changes if c.change_type.value == "added"]
        assert len(added) == 1

    def test_removed_step(self):
        pipe_a = PipelineSpec(steps=[PipelineStep(name="load", type="data"), PipelineStep(name="train", type="ml")])
        pipe_b = PipelineSpec(steps=[PipelineStep(name="load", type="data")])
        changes = diff_pipeline(pipe_a, pipe_b)
        removed = [c for c in changes if c.change_type.value == "removed"]
        assert len(removed) == 1


class TestSemanticDiff:
    def test_metrics_identical(self):
        changes = diff_metrics({"acc": 0.9, "loss": 0.1}, {"acc": 0.9, "loss": 0.1})
        assert len(changes) == 0

    def test_metrics_changed(self):
        changes = diff_metrics({"acc": 0.9}, {"acc": 0.85})
        assert len(changes) == 1
        assert "↓" in changes[0].description

    def test_metric_added(self):
        changes = diff_metrics({"acc": 0.9}, {"acc": 0.9, "f1": 0.88})
        added = [c for c in changes if c.change_type.value == "added"]
        assert len(added) == 1


class TestEnvironmentDiff:
    def test_identical(self):
        env = EnvironmentInfo(os="Linux", python_version="3.11")
        changes = diff_environment(env, env)
        assert len(changes) == 0

    def test_package_added(self):
        env_a = EnvironmentInfo(packages={"numpy": "1.24"})
        env_b = EnvironmentInfo(packages={"numpy": "1.24", "torch": "2.0"})
        changes = diff_environment(env_a, env_b)
        assert len(changes) == 1
        assert changes[0].change_type.value == "added"


class TestDiffEngine:
    def test_full_diff(self):
        exp_a = ExperimentIR(
            name="run-a",
            parameters={"lr": 0.01, "batch_size": 32},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.92}),
        )
        exp_b = ExperimentIR(
            name="run-b",
            parameters={"lr": 0.001, "batch_size": 64},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.88}),
        )
        engine = DiffEngine()
        report = engine.diff(exp_a, exp_b)
        assert len(report.changes) >= 2
        assert any("accuracy" in c.key for c in report.changes) or any("accuracy" in c.description for c in report.changes)


class TestCausalAttribution:
    def test_explain(self):
        from eide.causal.attribution import CausalAttribution

        exp_a = ExperimentIR(
            name="baseline",
            parameters={"lr": 0.01},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.92}),
        )
        exp_b = ExperimentIR(
            name="changed",
            parameters={"lr": 0.001},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.88}),
        )
        causal = CausalAttribution()
        explanation = causal.explain(exp_a, exp_b)
        assert "likely_cause" in explanation
        assert len(explanation["likely_cause"]) > 0
