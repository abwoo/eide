import tempfile
from pathlib import Path

from click.testing import CliRunner

from eide.cli.main import cli
from eide.core.ir import ExperimentIR
from eide.storage.filestore import FileStore


class TestProjectFlag:
    def test_create_with_project(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, ["create", "--name", "proj-test", "--project", "my-project", "--root", tmp])
            assert result.exit_code == 0
            store = FileStore(tmp)
            entries = store.list_experiments()
            assert len(entries) == 1
            exp = store.load(entries[0]["id"])
            assert exp.project == "my-project"

    def test_list_filter_by_project(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = ExperimentIR(name="a", project="p1")
            b = ExperimentIR(name="b", project="p2")
            store.save(a)
            store.save(b)

            result = runner.invoke(cli, ["list", "--project", "p1", "--root", tmp])
            assert result.exit_code == 0
            assert "p1" in result.output
            assert "p2" not in result.output  # filtered to p1 only

    def test_capture_manual_with_project(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, [
                "capture", "manual", "--name", "manual-proj", "--project", "p1",
                "--param", "lr=0.01", "--metric", "acc=0.9", "--root", tmp,
            ])
            assert result.exit_code == 0
            store = FileStore(tmp)
            for e in store.list_experiments():
                if e["name"] == "manual-proj":
                    exp = store.load(e["id"])
                    assert exp.project == "p1"
                    return
            assert False, "Experiment not found"

    def test_show_displays_project(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            exp = ExperimentIR(name="show-proj", project="visible")
            store.save(exp)
            result = runner.invoke(cli, ["show", exp.id, "--root", tmp])
            assert "visible" in result.output


class TestDiffFlag:
    def test_create_with_diff(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            from eide.core.types import ExperimentOutputs
            a = ExperimentIR(name="base", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            store.save(a)

            result = runner.invoke(cli, [
                "create", "--name", "variant", "--diff", "--root", tmp,
            ])
            assert result.exit_code == 0
            assert "Auto-diff" in result.output or "No baseline" in result.output

    def test_capture_manual_with_diff(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(tmp)
            a = ExperimentIR(name="base", parameters={"lr": 0.01}, outputs__metrics={"acc": 0.9})
            store.save(a)

            result = runner.invoke(cli, [
                "capture", "manual", "--name", "v2", "--param", "lr=0.02",
                "--metric", "acc=0.85", "--diff", "--root", tmp,
            ])
            assert result.exit_code == 0
            assert "Auto-diff" in result.output or "No baseline" in result.output


class TestContextProject:
    def test_context_manager_with_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            from eide.capture.context import ExperimentContext
            root = Path(tmp)
            with ExperimentContext("ctx-proj", project="ctx-project", store_root=str(root), auto_save=True, capture_env=False) as logger:
                logger.log_param("x", 1)
            store = FileStore(root)
            for e in store.list_experiments():
                if e["name"] == "ctx-proj":
                    exp = store.load(e["id"])
                    assert exp.project == "ctx-project"
                    return
            assert False

    def test_context_manager_with_resource_monitoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            from eide.capture.context import ExperimentContext
            root = Path(tmp)
            with ExperimentContext("resource-test", store_root=str(root), auto_save=True, capture_env=False, monitor_resources=True, resource_interval=0.1) as _:
                import time
                time.sleep(0.3)
            store = FileStore(root)
            for e in store.list_experiments():
                exp = store.load(e["id"])
                if exp.metadata and exp.metadata.get("resource_snapshots"):
                    snaps = exp.metadata["resource_snapshots"]
                    assert len(snaps) >= 1
                    return
            assert False, "No resource snapshots found"


class TestDashboardDetail:
    def test_experiment_detail_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            from httpx import ASGITransport, AsyncClient

            from eide.dashboard.app import create_app
            store = FileStore(tmp)
            from eide.core.types import ExperimentOutputs
            exp = ExperimentIR(name="detail-test", project="test-proj", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            store.save(exp)
            app = create_app(store)
            import anyio

            async def _run():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get(f"/experiments/{exp.id}")
                    assert resp.status_code == 200
                    assert "detail-test" in resp.text
                    assert "test-proj" in resp.text
                    assert "lr" in resp.text
                    assert "0.01" in resp.text
                    assert "acc" in resp.text
            anyio.run(_run)
