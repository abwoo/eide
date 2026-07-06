import tempfile
from pathlib import Path

import pytest

from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.graph.builder import ExperimentGraph
from eide.graph.queries import GraphQuery
from eide.storage.filestore import FileStore


@pytest.fixture
def populated_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(tmp)
        e1 = ExperimentIR(
            name="baseline",
            parameters={"lr": 0.01, "batch_size": 32},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.92}),
            tags={"dataset": "cifar10", "model": "cnn"},
        )
        e2 = ExperimentIR(
            name="tweak-lr",
            parameters={"lr": 0.001, "batch_size": 32},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.88}),
            tags={"dataset": "cifar10", "replay_of": e1.id},
        )
        e3 = ExperimentIR(
            name="different",
            parameters={"lr": 0.01, "batch_size": 128},
            outputs=ExperimentOutputs(metrics={"accuracy": 0.95}),
            tags={"dataset": "imagenet"},
        )
        store.save(e1)
        store.save(e2)
        store.save(e3)
        yield store


class TestExperimentGraphBuilder:
    def test_build(self, populated_store):
        eg = ExperimentGraph.build(populated_store)
        assert eg.graph.number_of_nodes() > 0
        assert eg.graph.number_of_edges() > 0

    def test_build_contains_experiments(self, populated_store):
        eg = ExperimentGraph.build(populated_store)
        entries = populated_store.list_experiments()
        for e in entries:
            assert eg.graph.has_node(e["id"])

    def test_parameter_nodes(self, populated_store):
        eg = ExperimentGraph.build(populated_store)
        param_nodes = [
            n for n, d in eg.graph.nodes(data=True)
            if d.get("type") == ExperimentGraph.NODE_PARAMETER
        ]
        assert len(param_nodes) >= 3

    def test_metric_nodes(self, populated_store):
        eg = ExperimentGraph.build(populated_store)
        metric_nodes = [
            n for n, d in eg.graph.nodes(data=True)
            if d.get("type") == ExperimentGraph.NODE_METRIC
        ]
        assert len(metric_nodes) >= 3

    def test_derived_edge(self, populated_store):
        eg = ExperimentGraph.build(populated_store)
        entries = populated_store.list_experiments()
        derived = entries[1]["id"]
        edges = list(eg.graph.edges(derived, data=True))
        derived_edges = [e for e in edges if e[2].get("type") == ExperimentGraph.EDGE_DERIVED_FROM]
        assert len(derived_edges) >= 1

    def test_save_load(self, populated_store):
        eg = ExperimentGraph.build(populated_store)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            eg.save(path)
            assert path.exists()
            loaded = ExperimentGraph.load(path)
            assert loaded.graph.number_of_nodes() == eg.graph.number_of_nodes()


class TestGraphQuery:
    @pytest.fixture
    def graph(self, populated_store):
        return ExperimentGraph.build(populated_store)

    def test_lineage(self, graph, populated_store):
        entries = populated_store.list_experiments()
        q = GraphQuery(graph)
        chain = q.lineage(entries[1]["id"])
        assert len(chain) >= 2
        assert chain[1]["id"] == entries[0]["id"]

    def test_most_used_parameters(self, graph):
        q = GraphQuery(graph)
        results = q.most_used_parameters(top_n=5)
        assert len(results) > 0
        for r in results:
            assert "key" in r
            assert "count" in r

    def test_experiments_by_parameter(self, graph):
        q = GraphQuery(graph)
        results = q.experiments_by_parameter("lr")
        assert len(results) > 0
        for r in results:
            assert "id" in r
            assert "param_value" in r

    def test_most_volatile_parameters(self, graph):
        q = GraphQuery(graph)
        results = q.most_volatile_parameters(top_n=5)
        assert len(results) > 0
        lr_entry = next((r for r in results if r["key"] == "lr"), None)
        assert lr_entry is not None
        assert lr_entry["unique_values"] >= 2
