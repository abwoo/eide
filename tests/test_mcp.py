from __future__ import annotations

import json
import tempfile
from pathlib import Path

import anyio

from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.diff.engine import DiffEngine
from eide.storage.filestore import FileStore

from eide.mcp.server import (
    _do_capture,
    _do_diff,
    _do_search,
    _do_list,
    _do_show,
    _do_delete,
    _do_explain,
    _do_replay,
    _do_graph_param_impact,
    _do_graph_duplicates,
    _do_graph_conflicts,
    _do_graph_lineage,
    _do_graph_top_params,
    _do_advise_risk,
    _do_advise_baseline,
)


def _make_store():
    tmp = Path(tempfile.mkdtemp())
    return FileStore(tmp)


def _seed(store, **overrides):
    params = {"name": "test-exp", "parameters": {"lr": 0.01}, "outputs": ExperimentOutputs(metrics={"acc": 0.9})}
    params.update(overrides)
    exp = ExperimentIR(**params)
    store.save(exp)
    return exp


class TestMCPCapture:
    def _check(self, **kwargs):
        store = _make_store()
        async def run():
            return await _do_capture(store, **kwargs)
        result = anyio.run(run)
        data = json.loads(result)
        assert "id" in data
        loaded = store.load(data["id"])
        return loaded, data

    def test_capture_basic(self):
        loaded, _ = self._check(name="mcp-test")
        assert loaded.name == "mcp-test"

    def test_capture_with_metrics(self):
        loaded, _ = self._check(name="metrics-test", metrics='{"acc": 0.95, "loss": 0.05}')
        assert loaded.outputs.metrics["acc"] == 0.95
        assert loaded.outputs.metrics["loss"] == 0.05

    def test_capture_with_pipeline_steps(self):
        loaded, _ = self._check(name="pipe-test", pipeline_steps='[{"name": "train"}, {"name": "eval"}]')
        assert len(loaded.pipeline.steps) == 2
        assert loaded.pipeline.steps[0].name == "train"

    def test_capture_with_tags(self):
        loaded, _ = self._check(name="tag-test", tags='{"env": "prod", "user": "alice"}')
        assert loaded.tags["env"] == "prod"
        assert loaded.tags["user"] == "alice"

    def test_capture_defaults(self):
        loaded, _ = self._check()
        assert loaded.name == ""
        assert loaded.parameters == {}
        assert loaded.outputs.metrics == {}


class TestMCPDiff:
    def test_diff_experiments(self):
        store = _make_store()
        a = _seed(store, name="a", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
        b = _seed(store, name="b", parameters={"lr": 0.02}, outputs=ExperimentOutputs(metrics={"acc": 0.85}))
        engine = DiffEngine()
        async def run():
            return await _do_diff(store, engine, a.id, b.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert "changes" in data
        assert len(data["changes"]) > 0

    def test_diff_not_found(self):
        store = _make_store()
        engine = DiffEngine()
        async def run():
            return await _do_diff(store, engine, "nonexistent", "nonexistent2")
        result = anyio.run(run)
        data = json.loads(result)
        assert "error" in data

    def test_diff_same_experiment(self):
        store = _make_store()
        a = _seed(store, name="same")
        engine = DiffEngine()
        async def run():
            return await _do_diff(store, engine, a.id, a.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert len(data["changes"]) == 0


class TestMCPSearch:
    def test_search_by_name(self):
        store = _make_store()
        _seed(store, name="unique-name")
        async def run():
            return await _do_search(store, "unique-name")
        result = anyio.run(run)
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["name"] == "unique-name"

    def test_search_all(self):
        store = _make_store()
        _seed(store, name="exp-a", parameters={"lr": 0.01})
        _seed(store, name="exp-b", parameters={"lr": 0.02})
        async def run():
            return await _do_search(store, "exp-")
        result = anyio.run(run)
        data = json.loads(result)
        assert len(data) == 2

    def test_search_no_results(self):
        store = _make_store()
        async def run():
            return await _do_search(store, "zzznotfound")
        result = anyio.run(run)
        data = json.loads(result)
        assert data == []


class TestMCPList:
    def test_list_empty(self):
        store = _make_store()
        async def run():
            return await _do_list(store)
        result = anyio.run(run)
        data = json.loads(result)
        assert data == []

    def test_list_with_experiments(self):
        store = _make_store()
        _seed(store, name="first")
        _seed(store, name="second")
        async def run():
            return await _do_list(store)
        result = anyio.run(run)
        data = json.loads(result)
        assert len(data) == 2
        names = {e["name"] for e in data}
        assert names == {"first", "second"}


class TestMCPShow:
    def test_show_experiment(self):
        store = _make_store()
        exp = _seed(store, name="show-me")
        async def run():
            return await _do_show(store, exp.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert data["name"] == "show-me"
        assert data["id"] == exp.id

    def test_show_not_found(self):
        store = _make_store()
        async def run():
            return await _do_show(store, "missing")
        result = anyio.run(run)
        data = json.loads(result)
        assert "error" in data


class TestMCPDelete:
    def test_delete_experiment(self):
        store = _make_store()
        exp = _seed(store, name="to-delete")
        async def run():
            return await _do_delete(store, exp.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert data["status"] == "deleted"
        assert store.list_experiments() == []

    def test_delete_nonexistent(self):
        store = _make_store()
        async def run():
            return await _do_delete(store, "ghost")
        result = anyio.run(run)
        data = json.loads(result)
        assert "error" in data


class TestMCPExplain:
    def test_explain_different(self):
        store = _make_store()
        a = _seed(store, name="a", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
        b = _seed(store, name="b", parameters={"lr": 0.02}, outputs=ExperimentOutputs(metrics={"acc": 0.85}))
        engine = DiffEngine()
        async def run():
            return await _do_explain(store, engine, a.id, b.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert "experiment_a" in data
        assert "experiment_b" in data
        assert "likely_cause" in data
        assert "impact_estimates" in data

    def test_explain_identical(self):
        store = _make_store()
        a = _seed(store, name="same")
        engine = DiffEngine()
        async def run():
            return await _do_explain(store, engine, a.id, a.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert data["likely_cause"] == "No significant changes detected"

    def test_explain_not_found(self):
        store = _make_store()
        engine = DiffEngine()
        async def run():
            return await _do_explain(store, engine, "nope", "nope2")
        result = anyio.run(run)
        data = json.loads(result)
        assert "error" in data


class TestMCPAdviseRisk:
    def test_risk_exists(self):
        store = _make_store()
        exp = _seed(store, name="risky")
        async def run():
            return await _do_advise_risk(store, exp.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert "risk_level" in data
        assert "risks" in data

    def test_risk_not_found(self):
        store = _make_store()
        async def run():
            return await _do_advise_risk(store, "missing")
        result = anyio.run(run)
        data = json.loads(result)
        assert "error" in data


class TestMCPGraphParamImpact:
    def test_param_impact(self):
        store = _make_store()
        _seed(store, name="impact-test", parameters={"lr": 0.01}, metrics={"acc": 0.95})
        from eide.graph.builder import ExperimentGraph
        ExperimentGraph.build(store)
        async def run():
            return await _do_graph_param_impact(store, "lr")
        result = anyio.run(run)
        data = json.loads(result)
        assert isinstance(data, list)

    def test_param_impact_empty(self):
        store = _make_store()
        async def run():
            return await _do_graph_param_impact(store, "missing_param")
        result = anyio.run(run)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 0


class TestMCPReplay:
    def test_replay_simple(self):
        store = _make_store()
        exp = _seed(store, name="replay-me", parameters={"lr": 0.01})
        async def run():
            return await _do_replay(store, exp.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert "replay_id" in data
        assert "summary" in data

    def test_replay_not_found(self):
        store = _make_store()
        async def run():
            return await _do_replay(store, "missing")
        result = anyio.run(run)
        data = json.loads(result)
        assert "error" in data


class TestMCPAdviseBaseline:
    def test_baseline_single(self):
        store = _make_store()
        a = _seed(store, name="a", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
        async def run():
            return await _do_advise_baseline(store, a.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert data["target_id"] == a.id

    def test_baseline_not_found(self):
        store = _make_store()
        async def run():
            return await _do_advise_baseline(store, "missing")
        result = anyio.run(run)
        data = json.loads(result)
        assert "error" in data


class TestMCPGraphDuplicates:
    def test_duplicates_none(self):
        store = _make_store()
        exp = _seed(store, name="unique")
        async def run():
            return await _do_graph_duplicates(store, exp.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert isinstance(data, list)

    def test_duplicates_not_found(self):
        store = _make_store()
        async def run():
            return await _do_graph_duplicates(store, "missing")
        result = anyio.run(run)
        data = json.loads(result)
        assert isinstance(data, list)


class TestMCPGraphConflicts:
    def test_conflicts_none(self):
        store = _make_store()
        exp = _seed(store, name="conflict-test")
        async def run():
            return await _do_graph_conflicts(store, exp.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert isinstance(data, list)

    def test_conflicts_not_found(self):
        store = _make_store()
        async def run():
            return await _do_graph_conflicts(store, "missing")
        result = anyio.run(run)
        data = json.loads(result)
        assert isinstance(data, list)


class TestMCPGraphLineage:
    def test_lineage_none(self):
        store = _make_store()
        exp = _seed(store, name="lineage-test")
        async def run():
            return await _do_graph_lineage(store, exp.id)
        result = anyio.run(run)
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) >= 1  # at least self

    def test_lineage_not_found(self):
        store = _make_store()
        async def run():
            return await _do_graph_lineage(store, "missing")
        result = anyio.run(run)
        data = json.loads(result)
        assert "error" in data or isinstance(data, list)


class TestMCPGraphTopParams:
    def test_top_params_empty(self):
        store = _make_store()
        _seed(store, name="no-params")
        async def run():
            return await _do_graph_top_params(store, top_n=5)
        result = anyio.run(run)
        data = json.loads(result)
        assert isinstance(data, list)

    def test_top_params_with_data(self):
        store = _make_store()
        _seed(store, name="with-params", parameters={"lr": 0.01, "bs": 64})
        async def run():
            return await _do_graph_top_params(store, top_n=10)
        result = anyio.run(run)
        data = json.loads(result)
        assert len(data) >= 1
