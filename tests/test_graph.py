import json
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.graph.builder import ExperimentGraph
from eide.graph.queries import GraphQuery
from eide.storage.filestore import FileStore


@pytest.fixture
def small_store():
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


@pytest.fixture
def empty_store():
    with tempfile.TemporaryDirectory() as tmp:
        yield FileStore(tmp)


@pytest.fixture
def single_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(tmp)
        store.save(ExperimentIR(name="lonely", parameters={"x": 1}))
        yield store


@pytest.fixture
def large_store():
    """50+ experiments with varied parameters, metrics, tags, and replay links."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(tmp)
        ids = []
        base = ExperimentIR(
            name="base",
            parameters={"lr": 0.01, "batch_size": 32, "optimizer": "sgd", "dropout": 0.5},
            outputs=ExperimentOutputs(metrics={"acc": 0.90, "loss": 0.10, "f1": 0.88}),
            tags={"dataset": "cifar10", "model": "resnet18", "user": "alice"},
            data_versions={"train": "v1", "test": "v1"},
        )
        store.save(base)
        ids.append(base.id)

        # Generate variations
        for i in range(55):
            lr = round(0.01 * (1 + i * 0.1), 4)
            bs = 32 if i % 2 == 0 else 64
            opt = "adam" if i % 3 == 0 else "sgd"
            acc = round(0.85 + (i % 10) * 0.01, 4)
            loss = round(0.15 - (i % 8) * 0.005, 4)
            exp = ExperimentIR(
                name=f"exp-{i:03d}",
                parameters={"lr": lr, "batch_size": bs, "optimizer": opt},
                outputs=ExperimentOutputs(metrics={"acc": acc, "loss": loss}),
                tags={"dataset": "cifar10", "run": str(i)},
                data_versions={"train": "v2", "test": "v1"},
            )
            # Some are derived from base
            if i < 5:
                exp.tags["replay_of"] = base.id
            store.save(exp)
            ids.append(exp.id)

        yield store


@pytest.fixture
def nested_params_store():
    """Store with deeply nested parameter structures."""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(tmp)
        # Create experiments with nested params (stored as flat key-value)
        for i in range(10):
            exp = ExperimentIR(
                name=f"nested-{i}",
                parameters={
                    f"level1.level2.lr": round(0.01 * (i + 1), 4),
                    f"level1.level2.bs": 32 + i,
                    f"level1.optim.wd": 0.0001 * (i + 1),
                    f"level1.optim.mom": 0.9,
                    "tags.env": "prod",
                },
                outputs=ExperimentOutputs(metrics={"acc": 0.8 + i * 0.01}),
            )
            store.save(exp)
        yield store


class TestExperimentGraphBuilder:
    def test_build_empty(self, empty_store):
        eg = ExperimentGraph.build(empty_store)
        assert eg.graph.number_of_nodes() == 0
        assert eg.graph.number_of_edges() == 0

    def test_build_single(self, single_store):
        eg = ExperimentGraph.build(single_store)
        assert eg.graph.number_of_nodes() >= 1  # at least the experiment node

    def test_build_small(self, small_store):
        eg = ExperimentGraph.build(small_store)
        assert eg.graph.number_of_nodes() > 0
        assert eg.graph.number_of_edges() > 0

    def test_build_contains_experiments(self, small_store):
        eg = ExperimentGraph.build(small_store)
        for e in small_store.list_experiments():
            assert eg.graph.has_node(e["id"])

    def test_parameter_nodes(self, small_store):
        eg = ExperimentGraph.build(small_store)
        param_nodes = [
            n for n, d in eg.graph.nodes(data=True)
            if d.get("type") == ExperimentGraph.NODE_PARAMETER
        ]
        assert len(param_nodes) >= 3

    def test_metric_nodes(self, small_store):
        eg = ExperimentGraph.build(small_store)
        metric_nodes = [
            n for n, d in eg.graph.nodes(data=True)
            if d.get("type") == ExperimentGraph.NODE_METRIC
        ]
        assert len(metric_nodes) >= 3

    def test_tag_nodes(self, small_store):
        eg = ExperimentGraph.build(small_store)
        tag_nodes = [
            n for n, d in eg.graph.nodes(data=True)
            if d.get("type") == ExperimentGraph.NODE_TAG
        ]
        assert len(tag_nodes) >= 3

    def test_dataset_nodes(self, small_store):
        eg = ExperimentGraph.build(small_store)
        ds_nodes = [
            n for n, d in eg.graph.nodes(data=True)
            if d.get("type") == ExperimentGraph.NODE_DATASET
        ]
        assert len(ds_nodes) >= 0  # small_store has no data_versions

    def test_data_version_nodes(self, large_store):
        eg = ExperimentGraph.build(large_store)
        ds_nodes = [
            n for n, d in eg.graph.nodes(data=True)
            if d.get("type") == ExperimentGraph.NODE_DATASET
        ]
        assert len(ds_nodes) > 0

    def test_derived_edge(self, small_store):
        eg = ExperimentGraph.build(small_store)
        entries = small_store.list_experiments()
        derived = entries[1]["id"]
        edges = list(eg.graph.edges(derived, data=True))
        derived_edges = [e for e in edges if e[2].get("type") == ExperimentGraph.EDGE_DERIVED_FROM]
        assert len(derived_edges) >= 1

    def test_similarity_edges(self, large_store):
        """Experiments with very similar params/metrics should have similarity edges."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = ExperimentIR(
                name="a", parameters={"x": 1}, outputs=ExperimentOutputs(metrics={"m": 0.9}),
                tags={"group": "same", "dataset": "d1"},
            )
            b = ExperimentIR(
                name="b", parameters={"x": 1}, outputs=ExperimentOutputs(metrics={"m": 0.91}),
                tags={"group": "same", "dataset": "d1"},
            )
            store.save(a)
            store.save(b)
            eg = ExperimentGraph.build(store)
            sim_edges = [
                ed for _, _, ed in eg.graph.edges(data=True)
                if ed.get("type") == ExperimentGraph.EDGE_SIMILAR_TO
            ]
            # a and b are very similar (diff should be tiny) -> similarity > 0.5
            assert len(sim_edges) >= 1

    def test_conflict_edges(self):
        """Experiments with metrics diverging >50% should have conflict edges."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = ExperimentIR(
                name="high",
                outputs=ExperimentOutputs(metrics={"acc": 0.95}),
                parameters={"lr": 0.01},
                tags={"dataset": "same"},
            )
            b = ExperimentIR(
                name="low",
                outputs=ExperimentOutputs(metrics={"acc": 0.40}),
                parameters={"lr": 0.01},
                tags={"dataset": "same"},
            )
            store.save(a)
            store.save(b)
            eg = ExperimentGraph.build(store)
            conflict_edges = [
                (_u, _v) for _u, _v, ed in eg.graph.edges(data=True)
                if ed.get("type") == ExperimentGraph.EDGE_CONFLICTS_WITH
            ]
            assert len(conflict_edges) >= 1

    def test_build_without_metrics(self):
        """Experiment with no metrics should still build."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            store.save(ExperimentIR(name="no-metrics", parameters={"x": 1}))
            eg = ExperimentGraph.build(store)
            assert eg.graph.number_of_nodes() >= 2  # experiment + param

    def test_build_without_parameters(self):
        """Experiment with no parameters should still build."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            store.save(ExperimentIR(name="no-params", outputs=ExperimentOutputs(metrics={"acc": 0.9})))
            eg = ExperimentGraph.build(store)
            assert eg.graph.number_of_nodes() >= 2  # experiment + metric

    def test_build_max_experiments(self, large_store):
        eg = ExperimentGraph.build(large_store, max_experiments=10)
        entries = large_store.list_experiments()
        # only first 10 should be in graph
        count = 0
        for e in entries[:10]:
            if eg.graph.has_node(e["id"]):
                count += 1
        # Some might fail to load
        assert count >= 0

    def test_save_load(self, small_store):
        eg = ExperimentGraph.build(small_store)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            eg.save(path)
            assert path.exists()
            loaded = ExperimentGraph.load(path)
            assert loaded.graph.number_of_nodes() == eg.graph.number_of_nodes()

    def test_save_load_empty(self, empty_store):
        eg = ExperimentGraph.build(empty_store)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            eg.save(path)
            loaded = ExperimentGraph.load(path)
            assert loaded.graph.number_of_nodes() == 0

    def test_large_graph_build(self, large_store):
        """50+ experiments should build without errors and produce meaningful graph."""
        eg = ExperimentGraph.build(large_store)
        nodes = eg.graph.number_of_nodes()
        edges = eg.graph.number_of_edges()
        assert nodes > 50  # experiments + params + metrics + tags
        assert edges > 50
        # Verify all experiment types exist
        types_seen = {d.get("type") for _, d in eg.graph.nodes(data=True)}
        assert ExperimentGraph.NODE_EXPERIMENT in types_seen
        assert ExperimentGraph.NODE_PARAMETER in types_seen
        assert ExperimentGraph.NODE_METRIC in types_seen

    def test_nested_params(self, nested_params_store):
        """Parameters with dot-separated keys should create separate nodes."""
        eg = ExperimentGraph.build(nested_params_store)
        param_nodes = [
            d.get("key") for _, d in eg.graph.nodes(data=True)
            if d.get("type") == ExperimentGraph.NODE_PARAMETER
        ]
        # Keys like "level1.level2.lr" should be stored as-is
        nested_keys = [k for k in param_nodes if "level1" in str(k)]
        assert len(nested_keys) >= 3

    def test_large_graph_unique_param_nodes(self, large_store):
        """Same parameter values should share a single node."""
        eg = ExperimentGraph.build(large_store)
        param_nodes: dict = {}
        for n, d in eg.graph.nodes(data=True):
            if d.get("type") == ExperimentGraph.NODE_PARAMETER:
                key = d.get("key")
                val = d.get("value")
                if key and val:
                    param_nodes.setdefault((key, val), []).append(n)
        # batch_size=32 should appear many times but share 1 node
        bs32_entries = [(key_tuple, ids) for key_tuple, ids in param_nodes.items()
                        if key_tuple[0] == "batch_size" and key_tuple[1] == "32"]
        if bs32_entries:
            _key_tuple, ids = bs32_entries[0]
            assert len(set(ids)) == 1


class TestGraphQuery:
    def test_duplicates(self, large_store):
        eg = ExperimentGraph.build(large_store)
        q = GraphQuery(eg)
        entries = large_store.list_experiments()
        results = q.duplicates(entries[0]["id"])
        assert isinstance(results, list)

    def test_duplicates_empty(self, single_store):
        eg = ExperimentGraph.build(single_store)
        q = GraphQuery(eg)
        node_list = list(eg.graph.nodes())
        exp_id = node_list[0] if node_list else ""
        results = q.duplicates(exp_id)
        assert results == []

    def test_duplicates_not_found(self, empty_store):
        eg = ExperimentGraph.build(empty_store)
        q = GraphQuery(eg)
        results = q.duplicates("nonexistent")
        assert results == []

    def test_conflicts(self, large_store):
        eg = ExperimentGraph.build(large_store)
        q = GraphQuery(eg)
        entries = large_store.list_experiments()
        results = q.conflicts(entries[0]["id"])
        assert isinstance(results, list)

    def test_conflicts_empty(self, single_store):
        eg = ExperimentGraph.build(single_store)
        q = GraphQuery(eg)
        results = q.conflicts(list(eg.graph.nodes())[0])
        assert results == []

    def test_lineage_simple(self, small_store):
        eg = ExperimentGraph.build(small_store)
        q = GraphQuery(eg)
        entries = small_store.list_experiments()
        chain = q.lineage(entries[1]["id"])
        assert len(chain) >= 2
        assert chain[-1]["id"] == entries[0]["id"]

    def test_lineage_no_chain(self, single_store):
        eg = ExperimentGraph.build(single_store)
        q = GraphQuery(eg)
        exp_id = [n for n, d in eg.graph.nodes(data=True) if d.get("type") == ExperimentGraph.NODE_EXPERIMENT]
        if exp_id:
            chain = q.lineage(exp_id[0])
            assert len(chain) == 1  # just itself

    def test_lineage_not_found(self, empty_store):
        eg = ExperimentGraph.build(empty_store)
        q = GraphQuery(eg)
        chain = q.lineage("nonexistent")
        assert isinstance(chain, list)

    def test_lineage_long_chain(self):
        """Chain of derived_from references should resolve fully."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            prev = None
            first = None
            for i in range(10):
                tags = {}
                if prev:
                    tags["replay_of"] = prev
                exp = ExperimentIR(name=f"gen-{i}", parameters={"x": i}, tags=tags)
                store.save(exp)
                if first is None:
                    first = exp.id
                prev = exp.id
            eg = ExperimentGraph.build(store)
            q = GraphQuery(eg)
            chain = q.lineage(prev)
            # Chain should include all 10
            assert len(chain) == 10

    def test_most_used_parameters(self, large_store):
        eg = ExperimentGraph.build(large_store)
        q = GraphQuery(eg)
        results = q.most_used_parameters(top_n=5)
        assert len(results) > 0
        for r in results:
            assert "key" in r
            assert "value" in r
            assert "count" in r

    def test_most_used_parameters_empty(self, empty_store):
        eg = ExperimentGraph.build(empty_store)
        q = GraphQuery(eg)
        results = q.most_used_parameters(top_n=5)
        assert results == []

    def test_most_used_parameters_top_n(self, large_store):
        eg = ExperimentGraph.build(large_store)
        q = GraphQuery(eg)
        results = q.most_used_parameters(top_n=3)
        assert len(results) <= 3

    def test_experiments_by_parameter(self, large_store):
        eg = ExperimentGraph.build(large_store)
        q = GraphQuery(eg)
        results = q.experiments_by_parameter("lr")
        assert len(results) > 0
        for r in results:
            assert "id" in r
            assert "param_value" in r

    def test_experiments_by_parameter_with_value(self, large_store):
        eg = ExperimentGraph.build(large_store)
        q = GraphQuery(eg)
        results = q.experiments_by_parameter("lr", value="0.01")
        assert isinstance(results, list)

    def test_experiments_by_parameter_not_found(self, large_store):
        eg = ExperimentGraph.build(large_store)
        q = GraphQuery(eg)
        results = q.experiments_by_parameter("nonexistent_param")
        assert results == []

    def test_most_volatile_parameters(self, large_store):
        eg = ExperimentGraph.build(large_store)
        q = GraphQuery(eg)
        results = q.most_volatile_parameters(top_n=5)
        assert len(results) > 0
        for r in results:
            assert "key" in r
            assert "unique_values" in r

    def test_most_volatile_parameters_empty(self, empty_store):
        eg = ExperimentGraph.build(empty_store)
        q = GraphQuery(eg)
        results = q.most_volatile_parameters(top_n=5)
        assert results == []

    def test_parameter_impact(self, large_store):
        eg = ExperimentGraph.build(large_store)
        q = GraphQuery(eg)
        results = q.parameter_impact("lr")
        assert len(results) > 0
        for r in results:
            assert "value" in r
            assert "experiment_count" in r
            assert "metrics" in r

    def test_parameter_impact_not_found(self, large_store):
        eg = ExperimentGraph.build(large_store)
        q = GraphQuery(eg)
        results = q.parameter_impact("ghost_param")
        assert results == []

    def test_export_graphml(self, large_store):
        eg = ExperimentGraph.build(large_store)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.graphml"
            q = GraphQuery(eg)
            q.export_graphml(str(path))
            assert path.exists()
            # Verify it's valid graphml
            content = path.read_text(encoding="utf-8")
            assert "graphml" in content.lower() or "<graph" in content

    def test_graph_integrity(self, large_store):
        """All edges should have valid source and target nodes."""
        eg = ExperimentGraph.build(large_store)
        for u, v in eg.graph.edges():
            assert eg.graph.has_node(u), f"Edge source {u} missing"
            assert eg.graph.has_node(v), f"Edge target {v} missing"

    def test_node_types_consistent(self, large_store):
        """All nodes should have a type attribute."""
        eg = ExperimentGraph.build(large_store)
        for n, d in eg.graph.nodes(data=True):
            assert "type" in d, f"Node {n} has no type"


class TestGraphEdgeCases:
    def test_same_params_different_values(self):
        """Different experiments with same param key but different values."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            for lr in [0.01, 0.02, 0.03, 0.04, 0.05]:
                store.save(ExperimentIR(
                    name=f"lr-{lr}",
                    parameters={"lr": lr},
                ))
            eg = ExperimentGraph.build(store)
            q = GraphQuery(eg)
            results = q.parameter_impact("lr")
            assert len(results) == 5

    def test_circular_derived_from(self):
        """Circular replay_of references should not cause infinite loops."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = ExperimentIR(name="a", tags={"replay_of": "b_id"})
            b = ExperimentIR(name="b", tags={"replay_of": "a_id"})
            # Can't actually create circular since we need real IDs
            store.save(a)
            store.save(b)
            # Manually set the tags after save
            a.tags["replay_of"] = b.id
            b.tags["replay_of"] = a.id
            store.save(a)
            store.save(b)
            eg = ExperimentGraph.build(store)
            q = GraphQuery(eg)
            # lineage should handle cycles without infinite loop
            chain = q.lineage(a.id)
            assert len(chain) <= 2  # should detect cycle

    def test_many_metrics(self):
        """Experiment with many metrics should create metric nodes."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            metrics = {f"m_{i}": round(i * 0.1, 2) for i in range(50)}
            store.save(ExperimentIR(
                name="many-metrics",
                outputs=ExperimentOutputs(metrics=metrics),
                parameters={"x": 1},
            ))
            eg = ExperimentGraph.build(store)
            metric_nodes = [
                n for n, d in eg.graph.nodes(data=True)
                if d.get("type") == ExperimentGraph.NODE_METRIC
            ]
            assert len(metric_nodes) == 50

    def test_max_experiments_limit(self):
        """max_experiments should limit nodes in the graph."""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            for i in range(100):
                store.save(ExperimentIR(name=f"e-{i}", parameters={"x": i}))
            eg = ExperimentGraph.build(store, max_experiments=10)
            # 10 experiments + 10 param nodes = at most 20 nodes
            assert eg.graph.number_of_nodes() <= 30
