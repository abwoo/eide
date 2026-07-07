from __future__ import annotations

import json
import tempfile
from pathlib import Path

import yaml
from click.testing import CliRunner

from eide.cli.main import cli
from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.storage.filestore import FileStore


def _seed(store, **overrides):
    params = {"name": "cli-test", "parameters": {"lr": 0.01}, "outputs": ExperimentOutputs(metrics={"acc": 0.9})}
    params.update(overrides)
    exp = ExperimentIR(**params)
    store.save(exp)
    return exp


class TestCLIInit:
    def test_init_creates_store(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, ["init", "--root", tmp])
            assert result.exit_code == 0
            assert "EIDE store" in result.output
            assert (Path(tmp) / "experiments").exists()

    def test_init_reinit(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, ["init", "--root", tmp])
            assert result.exit_code == 0
            result = runner.invoke(cli, ["init", "--root", tmp])
            assert result.exit_code == 0


class TestCLIDiff:
    def test_diff_text(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = _seed(store, name="a", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            b = _seed(store, name="b", parameters={"lr": 0.02}, outputs=ExperimentOutputs(metrics={"acc": 0.85}))
            result = runner.invoke(cli, ["diff", a.id, b.id, "--root", tmp])
            assert result.exit_code == 0
            assert "Diff Report" in result.output or "difference" in result.output

    def test_diff_json(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = _seed(store, name="a")
            b = _seed(store, name="b", parameters={"lr": 0.02})
            result = runner.invoke(cli, ["diff", a.id, b.id, "--root", tmp, "--format", "json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "changes" in data

    def test_diff_not_found(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, ["diff", "nonexistent", "nonexistent2", "--root", tmp])
            assert result.exit_code == 0
            assert "not found" in result.output.lower()


class TestCLIExplain:
    def test_explain_different(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = _seed(store, name="a", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            b = _seed(store, name="b", parameters={"lr": 0.02}, outputs=ExperimentOutputs(metrics={"acc": 0.85}))
            result = runner.invoke(cli, ["explain", a.id, b.id, "--root", tmp])
            assert result.exit_code == 0
            assert "expected" in result.output.lower() or "cause" in result.output.lower()

    def test_explain_not_found(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, ["explain", "nope", "nope2", "--root", tmp])
            assert result.exit_code == 0
            assert "not found" in result.output.lower()


class TestCLITag:
    def test_tag_add(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            exp = _seed(store)
            result = runner.invoke(cli, ["tag", "add", exp.id, "env", "prod", "--root", tmp])
            assert result.exit_code == 0
            assert "added" in result.output.lower()
            loaded = store.load(exp.id)
            assert loaded.tags["env"] == "prod"

    def test_tag_remove(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            exp = _seed(store)
            store.save(exp)
            runner.invoke(cli, ["tag", "add", exp.id, "env", "prod", "--root", tmp])
            result = runner.invoke(cli, ["tag", "remove", exp.id, "env", "--root", tmp])
            assert result.exit_code == 0
            assert "removed" in result.output.lower()

    def test_tag_list(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            exp = _seed(store)
            runner.invoke(cli, ["tag", "add", exp.id, "env", "prod", "--root", tmp])
            result = runner.invoke(cli, ["tag", "list", exp.id, "--root", tmp])
            assert result.exit_code == 0
            assert "prod" in result.output

    def test_tag_not_found(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, ["tag", "add", "missing", "k", "v", "--root", tmp])
            assert result.exit_code == 0
            assert "not found" in result.output.lower()


class TestCLIAdvise:
    def test_advise_compare(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = _seed(store, name="a", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            b = _seed(store, name="b", parameters={"lr": 0.02}, outputs=ExperimentOutputs(metrics={"acc": 0.85}))
            result = runner.invoke(cli, ["advise", "compare", a.id, b.id], env={"EIDE_ROOT": tmp})
            assert result.exit_code == 0
            assert "Comparison" in result.output

    def test_advise_risk(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            exp = _seed(store)
            result = runner.invoke(cli, ["advise", "risk", exp.id], env={"EIDE_ROOT": tmp})
            assert result.exit_code == 0
            assert "Risk" in result.output

    def test_advise_baseline(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = _seed(store, name="a", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            b = _seed(store, name="b", parameters={"lr": 0.02}, outputs=ExperimentOutputs(metrics={"acc": 0.85}))
            result = runner.invoke(cli, ["advise", "baseline", a.id], env={"EIDE_ROOT": tmp})
            assert result.exit_code == 0
            assert b.id in result.output


class TestCLIGraph:
    def test_graph_build(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            _seed(store)
            result = runner.invoke(cli, ["graph", "build", "--root", tmp])
            assert result.exit_code == 0
            assert "graph" in result.output.lower() or "node" in result.output.lower()

    def test_graph_duplicates(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = _seed(store, name="a", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            _seed(store, name="b", parameters={"lr": 0.02}, outputs=ExperimentOutputs(metrics={"acc": 0.85}))
            result = runner.invoke(cli, ["graph", "duplicates", a.id, "--root", tmp])
            assert result.exit_code == 0

    def test_graph_conflicts(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = _seed(store, name="a", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            result = runner.invoke(cli, ["graph", "conflicts", a.id, "--root", tmp])
            assert result.exit_code == 0

    def test_graph_lineage(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = _seed(store)
            result = runner.invoke(cli, ["graph", "lineage", a.id, "--root", tmp])
            assert result.exit_code == 0

    def test_graph_top_params(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            _seed(store, parameters={"lr": 0.01, "bs": 32})
            _seed(store, name="b", parameters={"lr": 0.02, "bs": 64})
            result = runner.invoke(cli, ["graph", "top-params", "--root", tmp, "--top", "5"])
            assert result.exit_code == 0


class TestCLIRun:
    def _write_config(self, tmp: str, config: dict) -> Path:
        path = Path(tmp) / "config.yml"
        path.write_text(yaml.dump(config), encoding="utf-8")
        return path

    def test_run_dry_run(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = {"name": "config-test", "parameters": {"lr": 0.01}, "metrics": {"acc": 0.95}}
            result = runner.invoke(cli, ["run", str(self._write_config(tmp, config)), "--root", tmp, "--dry-run"])
            assert result.exit_code == 0
            assert "Dry-run" in result.output
            assert "config-test" in result.output

    def test_run_dry_run_with_pipeline_list(self):
        """Pipeline as a list of step names."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = {"name": "pipeline-list", "pipeline": ["preprocess", "train", "eval"]}
            result = runner.invoke(cli, ["run", str(self._write_config(tmp, config)), "--root", tmp, "--dry-run"])
            assert result.exit_code == 0
            assert "3" in result.output  # 3 pipeline steps

    def test_run_dry_run_with_pipeline_dict(self):
        """Pipeline as a dict with 'steps' key (the format that caused the previous bug)."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = {"name": "pipeline-dict", "pipeline": {"name": "my-pipe", "steps": ["step1", "step2"]}}
            result = runner.invoke(cli, ["run", str(self._write_config(tmp, config)), "--root", tmp, "--dry-run"])
            assert result.exit_code == 0
            assert "2" in result.output  # 2 pipeline steps

    def test_run_dry_run_with_pipeline_step_objects(self):
        """Pipeline steps as dicts with name/type."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "name": "pipeline-objs",
                "pipeline": [
                    {"name": "load", "type": "data"},
                    {"name": "train", "type": "training"},
                ],
            }
            result = runner.invoke(cli, ["run", str(self._write_config(tmp, config)), "--root", tmp, "--dry-run"])
            assert result.exit_code == 0
            assert "2" in result.output

    def test_run_save(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = {"name": "config-test", "parameters": {"lr": 0.01}, "metrics": {"acc": 0.95}}
            result = runner.invoke(cli, ["run", str(self._write_config(tmp, config)), "--root", tmp])
            assert result.exit_code == 0
            assert "Created experiment" in result.output

    def test_run_save_and_load(self):
        """Verify saved experiment can be loaded back with correct data."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "name": "full-test",
                "project": "test-proj",
                "description": "a test run",
                "parameters": {"lr": 0.01, "bs": 32},
                "metrics": {"acc": 0.95, "loss": 0.05},
                "tags": {"model": "cnn", "dataset": "mnist"},
                "pipeline": ["load", "train", "eval"],
                "data_versions": {"train": "v2", "test": "v1"},
                "decisions": {"stop_early": "true"},
                "metadata": {"user": "alice"},
            }
            result = runner.invoke(cli, ["run", str(self._write_config(tmp, config)), "--root", tmp])
            assert result.exit_code == 0

            store = FileStore(tmp)
            entries = store.list_experiments()
            assert len(entries) == 1
            exp = store.load(entries[0]["id"])
            assert exp.name == "full-test"
            assert exp.project == "test-proj"
            assert exp.description == "a test run"
            assert exp.parameters == {"lr": 0.01, "bs": 32}
            assert exp.outputs.metrics == {"acc": 0.95, "loss": 0.05}
            assert exp.tags == {"model": "cnn", "dataset": "mnist"}
            assert len(exp.pipeline.steps) == 3
            assert exp.pipeline.steps[0].name == "load"
            assert exp.data_versions == {"train": "v2", "test": "v1"}
            assert exp.decisions == {"stop_early": "true"}
            assert exp.metadata == {"user": "alice"}

    def test_run_save_with_pipeline_dict(self):
        """The bug: pipeline as dict {name, steps} should extract 'steps' sub-key."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "name": "dict-pipe",
                "pipeline": {"name": "my-pipe", "steps": ["a", "b", "c"]},
            }
            result = runner.invoke(cli, ["run", str(self._write_config(tmp, config)), "--root", tmp])
            assert result.exit_code == 0

            store = FileStore(tmp)
            exp = store.load(store.list_experiments()[0]["id"])
            assert len(exp.pipeline.steps) == 3
            assert exp.pipeline.steps[0].name == "a"

    def test_run_with_diff_flag(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            from eide.core.ir import ExperimentIR
            from eide.core.types import ExperimentOutputs
            baseline = ExperimentIR(name="baseline", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            store.save(baseline)

            config = {"name": "diff-test", "parameters": {"lr": 0.02}, "metrics": {"acc": 0.85}}
            result = runner.invoke(cli, ["run", str(self._write_config(tmp, config)), "--root", tmp, "--diff"])
            assert result.exit_code == 0
            assert "Auto-diff" in result.output or "baseline" in result.output.lower()

    def test_run_empty_config(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = {}
            result = runner.invoke(cli, ["run", str(self._write_config(tmp, config)), "--root", tmp, "--dry-run"])
            assert result.exit_code == 0
            assert "Dry-run" in result.output

    def test_run_not_found(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "/nonexistent/config.yml"])
        assert result.exit_code != 0

    def test_run_save_then_list(self):
        """Run creates an experiment visible in list."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            config = {"name": "list-check", "project": "run-proj"}
            result = runner.invoke(cli, ["run", str(self._write_config(tmp, config)), "--root", tmp])
            assert result.exit_code == 0
            result = runner.invoke(cli, ["list", "--root", tmp])
            assert "list-check" in result.output
            assert "run-proj" in result.output


class TestCLIExportImport:
    def test_export(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            exp = _seed(store, name="export-me")
            out_path = Path(tmp) / "export.json"
            result = runner.invoke(cli, ["export", exp.id, str(out_path), "--root", tmp])
            assert result.exit_code == 0
            assert out_path.exists()
            data = json.loads(out_path.read_text(encoding="utf-8"))
            assert data["name"] == "export-me"

    def test_export_not_found(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.json"
            result = runner.invoke(cli, ["export", "missing", str(out_path), "--root", tmp])
            assert result.exit_code == 0
            assert "not found" in result.output.lower()

    def test_import(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            exp = _seed(store, name="roundtrip")
            out_path = Path(tmp) / "exp.json"
            runner.invoke(cli, ["export", exp.id, str(out_path), "--root", tmp])
            store.delete(exp.id)
            result = runner.invoke(cli, ["import-exp", str(out_path), "--root", tmp])
            assert result.exit_code == 0
            assert "Imported" in result.output
            entries = store.list_experiments()
            assert len(entries) == 1


class TestCLIList:
    def test_list_empty(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, ["list", "--root", tmp])
            assert result.exit_code == 0
            assert "No experiments" in result.output

    def test_list_with_data(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            _seed(store, name="alpha")
            _seed(store, name="beta")
            result = runner.invoke(cli, ["list", "--root", tmp])
            assert result.exit_code == 0
            assert "alpha" in result.output
            assert "beta" in result.output
