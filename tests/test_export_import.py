import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from eide.cli.export_import import export, import_exp
from eide.core.ir import ExperimentIR
from eide.storage.filestore import FileStore


class TestExportImport:
    def test_export_import_roundtrip(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            exp = ExperimentIR(name="test-exp", parameters={"lr": 0.01}, tags={"env": "prod"})
            store.save(exp)

            out_path = Path(tmp) / "export.json"

            result = runner.invoke(export, [exp.id, str(out_path), "--root", tmp])
            assert result.exit_code == 0
            assert out_path.exists()

            data = json.loads(out_path.read_text(encoding="utf-8"))
            assert data["name"] == "test-exp"
            assert data["parameters"]["lr"] == 0.01

            store2 = FileStore(Path(tmp) / "imported")
            result2 = runner.invoke(import_exp, [str(out_path), "--root", str(store2.root)])
            assert result2.exit_code == 0
            loaded = store2.load(exp.id)
            assert loaded.name == "test-exp"
            assert loaded.parameters["lr"] == 0.01
            assert loaded.tags["env"] == "prod"

    def test_export_not_found(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(export, ["nonexistent", str(Path(tmp) / "out.json"), "--root", tmp])
            assert result.exit_code == 0
            assert "not found" in result.output

    def test_import_with_override_id(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            exp = ExperimentIR(name="orig")
            store = FileStore(tmp)
            store.save(exp)

            out_path = Path(tmp) / "export.json"
            runner.invoke(export, [exp.id, str(out_path), "--root", tmp])

            store2 = FileStore(Path(tmp) / "imported")
            runner.invoke(import_exp, [str(out_path), "--root", str(store2.root), "--id", "override123"])
            loaded = store2.load("override123")
            assert loaded.name == "orig"


class TestFileStoreDiffWithBaseline:
    def test_diff_with_baseline_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = ExperimentIR(name="a", parameters={"lr": 0.01}, outputs__metrics={"acc": 0.9})
            b = ExperimentIR(name="b", parameters={"lr": 0.02}, outputs__metrics={"acc": 0.85})
            store.save(a)
            store.save(b)
            result = store.diff_with_baseline(b.id)
            assert result is not None
            assert "baseline_id" in result
            assert "change_count" in result
            assert result["change_count"] > 0

    def test_diff_with_baseline_no_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            exp = ExperimentIR(name="solo")
            store.save(exp)
            result = store.diff_with_baseline(exp.id)
            assert result is None

    def test_diff_with_baseline_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            result = store.diff_with_baseline("nonexistent")
            assert result is None
