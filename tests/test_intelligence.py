import tempfile

import pytest

from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.intelligence.advisor import ExperimentAdvisor
from eide.storage.filestore import FileStore


@pytest.fixture
def populated_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(tmp)
        e1 = ExperimentIR(
            name="baseline",
            parameters={"lr": 0.01, "batch_size": 32},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.92, "loss": 0.35}),
            tags={"dataset": "cifar10", "model": "cnn"},
        )
        e2 = ExperimentIR(
            name="tweak-lr",
            parameters={"lr": 0.001, "batch_size": 32},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.88, "loss": 0.42}),
            tags={"dataset": "cifar10", "replay_of": e1.id},
        )
        store.save(e1)
        store.save(e2)
        yield store


class TestExperimentAdvisor:
    def test_compare(self, populated_store):
        entries = populated_store.list_experiments()
        exp_a = populated_store.load(entries[0]["id"])
        exp_b = populated_store.load(entries[1]["id"])

        advisor = ExperimentAdvisor(store=populated_store)
        insight = advisor.compare(exp_a, exp_b)

        assert insight.experiment_a == entries[0]["id"]
        assert insight.experiment_b == entries[1]["id"]
        assert len(insight.likely_causes) > 0
        assert len(insight.recommended_actions) > 0

    def test_compare_no_changes(self, populated_store):
        entries = populated_store.list_experiments()
        exp = populated_store.load(entries[0]["id"])

        advisor = ExperimentAdvisor(store=populated_store)
        insight = advisor.compare(exp, exp)
        assert insight.diff_summary is not None

    def test_recommend_baseline(self, populated_store):
        entries = populated_store.list_experiments()
        exp = populated_store.load(entries[0]["id"])

        advisor = ExperimentAdvisor(store=populated_store)
        rec = advisor.recommend_baseline(exp)

        assert rec.target_id == entries[0]["id"]

    def test_assess_risk(self, populated_store):
        entries = populated_store.list_experiments()
        exp = populated_store.load(entries[0]["id"])

        advisor = ExperimentAdvisor(store=populated_store)
        assessment = advisor.assess_risk(exp)

        assert assessment.experiment_id == entries[0]["id"]
        assert assessment.risk_level in ("low", "medium", "high", "unknown")

    def test_assess_risk_replay(self, populated_store):
        entries = populated_store.list_experiments()
        exp = populated_store.load(entries[1]["id"])

        advisor = ExperimentAdvisor(store=populated_store)
        assessment = advisor.assess_risk(exp)

        replay_risks = [r for r in assessment.risks if r["type"] == "replay"]
        assert len(replay_risks) >= 1
