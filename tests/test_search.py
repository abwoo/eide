from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from eide.core.ir import ExperimentIR
from eide.storage.filestore import FileStore


@pytest.fixture
def store():
    tmp = Path(tempfile.mkdtemp())
    s = FileStore(tmp)
    for i in range(5):
        exp = ExperimentIR(
            name=f"exp-{i}",
            parameters={"lr": 0.001 * (i + 1), "batch_size": 32, "nested": {"dropout": 0.1 * i}},  # noqa: S311
            outputs__metrics={"accuracy": 0.8 + i * 0.04, "loss": 0.5 - i * 0.05},
            tags={"dataset": "mnist" if i % 2 == 0 else "cifar", "run": str(i)},
            metadata={"user": "alice"},
        )
        s.save(exp)
    return s


class TestSearch:
    def test_search_by_name(self, store: FileStore):
        results = store.search("exp-0", field="name")
        assert len(results) == 1
        assert results[0].name == "exp-0"

    def test_search_by_tag(self, store: FileStore):
        results = store.search("dataset:mnist", field="tag")
        assert len(results) == 3

    def test_search_by_param(self, store: FileStore):
        results = store.search("0.002", field="param")
        assert len(results) == 1

    def test_search_by_metadata(self, store: FileStore):
        results = store.search("alice", field="metadata")
        assert len(results) == 5

    def test_search_all_fields(self, store: FileStore):
        results = store.search("cifar", field="all")
        assert len(results) == 2

    def test_search_no_results(self, store: FileStore):
        results = store.search("nothing", field="all")
        assert len(results) == 0

    def test_search_wildcard(self, store: FileStore):
        results = store.search("exp-*", field="name")
        assert len(results) == 5

    def test_find_by_metadata(self, store: FileStore):
        results = store.find_by_metadata("user", "alice")
        assert len(results) == 5

    def test_search_by_id(self, store: FileStore):
        all_exps = store.list_experiments()
        target_id = all_exps[0]["id"]
        results = store.search(target_id[:8], field="id")
        assert len(results) == 1
        assert results[0].id == target_id

    def test_flatten_params_nested(self):
        from eide.storage.filestore import _flatten_params
        flat = _flatten_params({"a": {"b": 1, "c": 2}, "d": 3})
        assert flat == {"a.b": 1, "a.c": 2, "d": 3}


class TestCLISearch:
    def test_cli_search(self):
        from click.testing import CliRunner

        from eide.cli.search import search

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            exp = ExperimentIR(name="test-search", parameters={"lr": 0.01}, tags={"env": "prod"})
            store.save(exp)

            result = runner.invoke(search, ["test-search", "--root", tmp])
            assert result.exit_code == 0
            assert "test-search" in result.output

    def test_cli_search_json(self):
        from click.testing import CliRunner

        from eide.cli.search import search

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            store.save(ExperimentIR(name="json-test"))
            result = runner.invoke(search, ["json-test", "--json", "--root", tmp])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert len(data) == 1
            assert data[0]["name"] == "json-test"
