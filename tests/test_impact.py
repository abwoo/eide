from __future__ import annotations

from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.diff.engine import DiffEngine


class TestImpactEstimate:
    def test_metric_correlation_with_param_change(self):
        a = ExperimentIR(
            name="base",
            parameters={"lr": 0.001, "batch_size": 32},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.85}),
        )
        b = ExperimentIR(
            name="changed",
            parameters={"lr": 0.01, "batch_size": 32},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.91}),
        )
        engine = DiffEngine()
        report = engine.diff(a, b)
        assert len(report.impact_estimates) == 1
        ie = report.impact_estimates[0]
        assert ie.metric == "accuracy"
        assert ie.likely_cause != ""
        assert "lr" in ie.likely_cause.lower() or "learning" in ie.likely_cause.lower()
        assert ie.confidence > 0.3

    def test_metric_correlation_with_env_change(self):
        from eide.core.types import EnvironmentInfo
        base_env = EnvironmentInfo(python_version="3.9", packages={"numpy": "1.21"}, hardware={})
        changed_env = EnvironmentInfo(python_version="3.11", packages={"numpy": "1.26"}, hardware={})
        a = ExperimentIR(
            name="base",
            parameters={"lr": 0.001},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.85}),
            environment=base_env,
        )
        b = ExperimentIR(
            name="changed",
            parameters={"lr": 0.001},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.90}),
            environment=changed_env,
        )
        engine = DiffEngine()
        report = engine.diff(a, b)
        assert len(report.impact_estimates) == 1
        ie = report.impact_estimates[0]
        assert ie.confidence > 0.05

    def test_no_metric_changes(self):
        a = ExperimentIR(name="base", parameters={"lr": 0.001})
        b = ExperimentIR(name="changed", parameters={"lr": 0.01})
        engine = DiffEngine()
        report = engine.diff(a, b)
        assert len(report.impact_estimates) == 0

    def test_direction_increase(self):
        a = ExperimentIR(name="a", outputs=ExperimentOutputs(metrics={"acc": 0.5}))
        b = ExperimentIR(name="b", outputs=ExperimentOutputs(metrics={"acc": 0.8}))
        engine = DiffEngine()
        report = engine.diff(a, b)
        assert report.impact_estimates[0].change_direction == "increase"

    def test_direction_decrease(self):
        a = ExperimentIR(name="a", outputs=ExperimentOutputs(metrics={"acc": 0.8}))
        b = ExperimentIR(name="b", outputs=ExperimentOutputs(metrics={"acc": 0.5}))
        engine = DiffEngine()
        report = engine.diff(a, b)
        assert report.impact_estimates[0].change_direction == "decrease"
