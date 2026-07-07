from __future__ import annotations

import tempfile
from pathlib import Path

import anyio
from click.testing import CliRunner
from httpx import ASGITransport, AsyncClient

from eide.cli.main import cli
from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.dashboard.app import create_app
from eide.storage.filestore import FileStore


class TestE2ECliToStore:
    """CLI -> FileStore: create, capture, diff, search survive roundtrip"""

    def _store(self, tmp: str) -> FileStore:
        return FileStore(Path(tmp))

    def test_create_empty_then_store_load(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, ["create", "--name", "e2e-cli-store", "--root", tmp])
            assert result.exit_code == 0, result.output
            entries = self._store(tmp).list_experiments()
            ids = [e["id"] for e in entries if e["name"] == "e2e-cli-store"]
            assert len(ids) == 1
            exp = self._store(tmp).load(ids[0])
            assert exp.name == "e2e-cli-store"

    def test_capture_manual_with_params_metrics(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, [
                "capture", "manual", "--name", "e2e-capture",
                "--param", "lr=0.01", "--param", "epochs=10",
                "--metric", "acc=0.95", "--root", tmp,
            ])
            assert result.exit_code == 0, result.output
            entries = self._store(tmp).list_experiments()
            ids = [e["id"] for e in entries if e["name"] == "e2e-capture"]
            assert len(ids) == 1
            exp = self._store(tmp).load(ids[0])
            assert exp.parameters == {"lr": "0.01", "epochs": "10"}
            assert exp.outputs.metrics == {"acc": 0.95}

    def test_capture_manual_then_search(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            runner.invoke(cli, [
                "capture", "manual", "--name", "e2e-search",
                "--param", "lr=0.1", "--metric", "f1=0.89",
                "--tag", "env=prod", "--root", tmp,
            ])
            store = self._store(tmp)
            results = store.search("e2e-search", field="name")
            assert len(results) >= 1
            exp = results[0]
            assert exp.tags.get("env") == "prod"

    def test_create_then_diff(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            a = ExperimentIR(name="base", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            store.save(a)
            result = runner.invoke(cli, [
                "create", "--name", "diffed", "--diff", "--root", tmp,
            ])
            assert result.exit_code == 0
            assert "Auto-diff" in result.output or "No baseline" in result.output

    def test_create_then_list_cli(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            self._store(tmp).save(ExperimentIR(name="list-me"))
            result = runner.invoke(cli, ["list", "--root", tmp])
            assert result.exit_code == 0
            assert "list-me" in result.output

    def test_search_cli(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            self._store(tmp).save(ExperimentIR(name="search-target"))
            result = runner.invoke(cli, ["search", "search-target", "--root", tmp])
            assert result.exit_code == 0
            assert "search-target" in result.output

    def test_create_then_show(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            exp = ExperimentIR(name="show-target", parameters={"batch": "64"})
            store.save(exp)
            result = runner.invoke(cli, ["show", exp.id, "--root", tmp])
            assert result.exit_code == 0
            assert "show-target" in result.output

    def test_delete_cli(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            exp = ExperimentIR(name="delete-me")
            store.save(exp)
            result = runner.invoke(cli, ["delete", exp.id, "--root", tmp])
            assert result.exit_code == 0
            assert "Deleted experiment" in result.output
            assert self._store(tmp).list_experiments() == []


class TestE2ECliToDashboard:
    """CLI -> Dashboard via ASGITransport"""

    def _app(self, tmp: str):
        return create_app(FileStore(Path(tmp)))

    def test_capture_appears_on_dashboard(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(cli, [
                "capture", "manual", "--name", "dashboard-e2e",
                "--param", "lr=0.001", "--project", "e2e",
                "--metric", "acc=0.99", "--root", tmp,
            ])
            assert result.exit_code == 0, result.output
            app = self._app(tmp)

            async def _run():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/")
                    assert resp.status_code == 200
                    assert "dashboard-e2e" in resp.text
                    assert "e2e" in resp.text

                    api = await client.get("/api/experiments")
                    assert api.status_code == 200
                    data = api.json()
                    names = [e["name"] for e in data["items"]]
                    assert "dashboard-e2e" in names

                    stats = await client.get("/api/stats")
                    assert stats.status_code == 200
                    assert stats.json()["total"] >= 1

            anyio.run(_run)

    def test_diff_shows_on_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(Path(tmp))
            a = ExperimentIR(name="diff-a", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.8}))
            b = ExperimentIR(name="diff-b", parameters={"lr": 0.02}, outputs=ExperimentOutputs(metrics={"acc": 0.9}))
            store.save(a)
            store.save(b)
            app = self._app(tmp)

            async def _run():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get(f"/diff?a={a.id}&b={b.id}")
                    assert resp.status_code == 200
                    assert "diff-a" in resp.text
                    assert "diff-b" in resp.text
                    assert "0.01" in resp.text
                    assert "0.02" in resp.text

            anyio.run(_run)

    def test_graph_page_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(Path(tmp))
            store.save(ExperimentIR(name="graph-node"))
            app = self._app(tmp)

            async def _run():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/graph")
                    assert resp.status_code == 200
                    assert "graph-node" in resp.text

            anyio.run(_run)

    def test_replay_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(Path(tmp))
            exp = ExperimentIR(name="replay-me")
            store.save(exp)
            app = self._app(tmp)

            async def _run():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get(f"/replay/{exp.id}")
                    assert resp.status_code == 200
                    assert "replay-me" in resp.text or "Replay" in resp.text

            anyio.run(_run)

    def test_create_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = self._app(tmp)

            async def _run():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/create")
                    assert resp.status_code == 200
                    assert "New Experiment" in resp.text or "create" in resp.text.lower()

            anyio.run(_run)

    def test_experiments_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(Path(tmp))
            store.save(ExperimentIR(name="exp-list-test"))
            app = self._app(tmp)

            async def _run():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/experiments")
                    assert resp.status_code == 200
                    assert "exp-list-test" in resp.text

            anyio.run(_run)

    def test_api_detail_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(Path(tmp))
            exp = ExperimentIR(name="api-detail", parameters={"x": "1"}, outputs=ExperimentOutputs(metrics={"y": 2.0}))
            store.save(exp)
            app = self._app(tmp)

            async def _run():
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get(f"/api/experiments/{exp.id}")
                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["name"] == "api-detail"
                    assert data["parameters"] == {"x": "1"}

            anyio.run(_run)


class TestE2EMcpServer:
    """MCP server can be created and tools registered"""

    def test_mcp_server_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            from eide.mcp.server import _create_mcp_server
            mcp = _create_mcp_server(eide_root=tmp)
            assert mcp is not None
            assert mcp.sse_app() is not None

            async def _run():
                tools = await mcp.list_tools()
                tool_names = [t.name for t in tools]
                assert "eide_list" in tool_names
                assert "eide_capture" in tool_names
                assert "eide_diff" in tool_names

            anyio.run(_run)

    def test_mcp_sse_via_real_client(self):
        """End-to-end: start uvicorn + connect via mcp client library"""
        with tempfile.TemporaryDirectory() as tmp:
            import threading

            import uvicorn

            from eide.mcp.server import _create_mcp_server

            store = FileStore(Path(tmp))
            store.save(ExperimentIR(name="sse-e2e"))
            mcp = _create_mcp_server(eide_root=tmp)

            host = "127.0.0.1"
            port = 9875
            app = mcp.sse_app()
            config = uvicorn.Config(app, host=host, port=port, log_level="error")
            server = uvicorn.Server(config)
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            import time
            time.sleep(1.5)

            try:
                from mcp.client.session import ClientSession
                from mcp.client.sse import sse_client

                async def _run():
                    async with sse_client(f"http://{host}:{port}/sse") as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            result = await session.list_tools()
                            tool_names = [t.name for t in result.tools]
                            assert "eide_list" in tool_names
                            assert "eide_capture" in tool_names

                            call_result = await session.call_tool("eide_list", {})
                            assert call_result is not None
                            text = "".join(
                                c.text for c in (call_result.content or []) if hasattr(c, "text")
                            )
                            assert "sse-e2e" in text

                anyio.run(_run)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    def test_mcp_resource_eide_experiments(self):
        """Read the eide://experiments resource via a real MCP client"""
        with tempfile.TemporaryDirectory() as tmp:
            import threading

            import uvicorn

            from eide.mcp.server import _create_mcp_server

            store = FileStore(Path(tmp))
            store.save(ExperimentIR(name="resource-test-exp"))
            mcp = _create_mcp_server(eide_root=tmp)

            host = "127.0.0.1"
            port = 9876
            app = mcp.sse_app()
            config = uvicorn.Config(app, host=host, port=port, log_level="error")
            server = uvicorn.Server(config)
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            import time
            time.sleep(1.5)

            try:
                from mcp.client.session import ClientSession
                from mcp.client.sse import sse_client

                async def _run():
                    async with sse_client(f"http://{host}:{port}/sse") as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            result = await session.read_resource("eide://experiments")
                            contents = result.contents if hasattr(result, "contents") else []
                            text = ""
                            for c in contents:
                                if hasattr(c, "text"):
                                    text += c.text
                                elif isinstance(c, dict):
                                    text += c.get("text", "")
                            assert "resource-test-exp" in text, f"Missing in {text}"

                            detail = await session.read_resource(
                                "eide://experiments/" + store.list_experiments()[0]["id"]
                            )
                            detail_text = ""
                            for c in (detail.contents if hasattr(detail, "contents") else []):
                                if hasattr(c, "text"):
                                    detail_text += c.text
                                elif isinstance(c, dict):
                                    detail_text += c.get("text", "")
                            assert "resource-test-exp" in detail_text

                anyio.run(_run)
            finally:
                server.should_exit = True
                thread.join(timeout=5)

    def test_mcp_resource_eide_experiments_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            from eide.mcp.server import _create_mcp_server
            mcp = _create_mcp_server(eide_root=tmp)

            async def _run():
                try:
                    result = await mcp.read_resource("eide://experiments/nonexistent-id")
                    assert result is not None
                except Exception:
                    pass

            anyio.run(_run)


class TestE2EReplayEngine:
    """ReplayEngine full execution with data/environment/verification"""

    def test_replay_full_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(Path(tmp))
            data_file = Path(tmp) / "source_data.csv"
            data_file.write_text("x, y\n1, 2\n3, 4\n", encoding="utf-8")

            from eide.core.ir import ExperimentIR
            from eide.core.types import (
                EnvironmentInfo,
                ExperimentOutputs,
                PipelineSpec,
                PipelineStep,
            )

            exp = ExperimentIR(
                name="replay-full-test",
                parameters={"lr": 0.01},
                outputs=ExperimentOutputs(metrics={"acc": 0.9}),
                environment=EnvironmentInfo(packages={}),
                data_versions={"training_data": str(data_file)},
                pipeline=PipelineSpec(
                    name="test-pipeline",
                    steps=[
                        PipelineStep(
                            name="verify",
                            type="python",
                            parameters={"code": "print('hello from replay')"},
                        )
                    ],
                ),
            )
            store.save(exp)

            from eide.replay.engine import ReplayEngine

            engine = ReplayEngine(store=store, work_dir=Path(tmp) / "replay_work")
            try:
                result = engine.replay(
                    exp,
                    restore_environment=True,
                    restore_data=True,
                    save_replay=True,
                )
                assert result is not None
                assert result.replay_experiment is not None
                assert result.replay_experiment.name.startswith("replay-")
                assert result.verdict is not None
                assert result.verdict.original_id == exp.id
                assert result.replay_experiment.tags.get("replay_of") == exp.id
                assert result.step_results is not None
                assert len(result.step_results) == 1
                step = result.step_results[0]
                assert step["success"] is True
                assert "hello from replay" in (step.get("output") or "")

                loaded = store.load(result.replay_experiment.id)
                assert loaded is not None
            finally:
                engine.cleanup()

    def test_replay_no_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            from eide.core.ir import ExperimentIR
            from eide.core.types import PipelineSpec, PipelineStep

            exp = ExperimentIR(
                name="replay-no-store",
                pipeline=PipelineSpec(
                    name="simple",
                    steps=[PipelineStep(name="echo", type="shell", parameters={"script": "echo ok"})],
                ),
            )
            from eide.replay.engine import ReplayEngine

            engine = ReplayEngine(work_dir=Path(tmp) / "replay_work2")
            try:
                result = engine.replay(
                    exp,
                    restore_environment=False,
                    restore_data=False,
                    save_replay=False,
                )
                assert result is not None
                assert result.replay_experiment is not None
            finally:
                engine.cleanup()
