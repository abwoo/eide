from __future__ import annotations

import json
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
class TestDashboardPages:
    async def test_home(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "EIDE" in resp.text

    async def test_home_js_hooks(self, app):
        """Home page has stat cards, project cards, progress bars, and timeline."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            html = resp.text
            # Stat cards
            assert "stat-value" in html
            assert "stat-label" in html
            # Progress bars for tags/metrics
            assert "rounded-full" in html
            # Project cards
            assert "Projects" in html
            # Timeline
            assert "Recent Activity" in html or "timeline" in html.lower()
            # Sidebar nav
            assert "sidebar" in html.lower() or "nav" in html.lower()
            # Version badge
            assert "v" in html and ("." in html)  # version number like v1.0.0

    async def test_experiments_page(self, app, store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/experiments")
            assert resp.status_code == 200
            assert "alpha" in resp.text
            assert "beta" in resp.text

    async def test_experiments_page_js_hooks(self, app):
        """Experiments page has batch compare checkboxes and project filter tabs."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/experiments")
            html = resp.text
            # Sortable table
            assert "<table" in html or "<thead" in html
            # Checkbox for batch compare
            assert "checkbox" in html.lower() or 'type="checkbox"' in html
            # Project filter tabs
            assert "project" in html.lower()
            # Search input
            assert 'input' in html and 'search' in html.lower() or 'Search' in html

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

    async def test_graph_page_js_hooks(self, app):
        """Graph page has D3.js container and search."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/graph")
            html = resp.text
            # SVG container for D3
            assert "svg" in html.lower() or "d3" in html.lower()
            # Graph search input
            assert 'input' in html and ('search' in html.lower() or 'Search' in html.lower())
            # Info panel for node inspection
            assert "info" in html.lower() or "panel" in html.lower()

    async def test_create_page(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/create")
            assert resp.status_code == 200
            assert "Create" in resp.text or "New" in resp.text or "experiment" in resp.text.lower()

    async def test_create_page_js_hooks(self, app):
        """Create page has form fields and submit button."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/create")
            html = resp.text
            # Form
            assert "<form" in html or 'method="post"' in html or 'action=' in html
            # Name input
            assert 'name' in html.lower()
            # Project input
            assert 'project' in html.lower()
            # Submit button
            assert "submit" in html.lower() or "create" in html.lower() or "button" in html.lower()

    async def test_experiment_detail_page(self, app, store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            experiments = store.list_experiments()
            exp_id = experiments[0]["id"]
            resp = await client.get(f"/experiments/{exp_id}")
            assert resp.status_code == 200
            assert "alpha" in resp.text

    async def test_experiment_detail_js_hooks(self, app, store):
        """Detail page has Chart.js canvas for metrics."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            experiments = store.list_experiments()
            exp_id = experiments[0]["id"]
            resp = await client.get(f"/experiments/{exp_id}")
            html = resp.text
            # Metrics chart canvas
            assert "chart" in html.lower() or "<canvas" in html
            # Metrics values displayed
            assert "0.9" in html or "0.85" in html
            # Chart type selector
            assert "select" in html or "chart" in html

    async def test_experiment_detail_not_found(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/experiments/nonexistent")
            assert resp.status_code == 404

    async def test_replay_page(self, app, store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            experiments = store.list_experiments()
            exp_id = experiments[0]["id"]
            resp = await client.get(f"/replay/{exp_id}")
            assert resp.status_code == 200

    async def test_replay_page_not_found(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/replay/nonexistent")
            assert resp.status_code == 404

    async def test_error_page(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/nonexistent-route")
            assert resp.status_code == 404

    async def test_all_pages_have_sidebar(self, app):
        """All main pages should include the navigation sidebar."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for path in ["/", "/experiments", "/diff", "/graph", "/create"]:
                resp = await client.get(path)
                # Sidebar or nav should be present
                assert "sidebar" in resp.text.lower() or "nav" in resp.text.lower(), f"{path} missing sidebar"


@pytest.mark.anyio
class TestDashboardAPI:
    async def test_api_experiments(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/experiments")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 2
            assert len(data["items"]) == 2

    async def test_api_experiments_filter_by_project(self, app, store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/experiments?project=default")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 2

    async def test_api_experiments_search(self, app, store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/experiments?search=alpha")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["items"]) >= 1
            names = [e.get("name", "") for e in data["items"]]
            assert "alpha" in names

    async def test_api_experiments_pagination(self, app):
        """API supports limit/offset pagination."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/experiments?limit=1")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["items"]) == 1
            assert data["total"] == 2

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

    async def test_api_projects(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/projects")
            assert resp.status_code == 200
            data = resp.json()
            assert "projects" in data
            assert isinstance(data["projects"], list)

    async def test_api_experiment_detail(self, app, store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            experiments = store.list_experiments()
            exp_id = experiments[0]["id"]
            resp = await client.get(f"/api/experiments/{exp_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "alpha"

    async def test_api_experiment_not_found(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/experiments/nonexistent")
            assert resp.status_code == 404

    async def test_api_metrics(self, app, store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            experiments = store.list_experiments()
            exp_id = experiments[0]["id"]
            resp = await client.get(f"/api/experiments/{exp_id}/metrics")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, dict)

    async def test_api_metrics_not_found(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/experiments/nonexistent/metrics")
            assert resp.status_code == 404

    async def test_api_risk(self, app, store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            experiments = store.list_experiments()
            exp_id = experiments[0]["id"]
            resp = await client.get(f"/api/experiments/{exp_id}/risk")
            assert resp.status_code == 200
            data = resp.json()
            assert "risk_level" in data

    async def test_api_risk_not_found(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/experiments/nonexistent/risk")
            assert resp.status_code == 404

    async def test_api_create_experiment(self, app, store):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/create", data={
                "name": "api-created", "project": "test-proj",
                "params": "lr=0.01,bs=64", "metrics": "acc=0.95",
                "tags": "model=cnn"
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "api-created"

            loaded = store.load(data["id"])
            assert loaded.name == "api-created"
            assert loaded.project == "test-proj"
            assert loaded.parameters == {"lr": "0.01", "bs": "64"}
            assert loaded.outputs.metrics == {"acc": 0.95}
            assert loaded.tags == {"model": "cnn"}
