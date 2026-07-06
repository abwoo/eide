from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.dashboard.app import create_app
from eide.storage.filestore import FileStore


@pytest.fixture
def store():
    tmp = Path(tempfile.mkdtemp())
    s = FileStore(tmp)
    s.save(ExperimentIR(name="alpha", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9})))
    s.save(ExperimentIR(name="beta", parameters={"lr": 0.02}, outputs=ExperimentOutputs(metrics={"acc": 0.85})))
    return s


@pytest.fixture
def app(store):
    return create_app(store)


@pytest.mark.anyio
class TestDashboard:
    async def test_home(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "EIDE" in resp.text

    async def test_experiments_page(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/experiments")
            assert resp.status_code == 200
            assert "alpha" in resp.text
            assert "beta" in resp.text

    async def test_diff_page(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/diff")
            assert resp.status_code == 200

    async def test_graph_page(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/graph")
            assert resp.status_code == 200
            assert "Knowledge Graph" in resp.text

    async def test_api_experiments(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/experiments")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2

    async def test_api_graph(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/graph")
            assert resp.status_code == 200
            data = resp.json()
            assert "nodes" in data
            assert "edges" in data

    async def test_api_stats(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 2
            assert "acc" in data["metric_range"]
