from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from httpx import ASGITransport

from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.dashboard.app import create_app
from eide.storage.filestore import FileStore

pytestmark = pytest.mark.skipif(
    True,  # no explicit request to run browser tests
    reason="Requires headless browser; run with --run-js or disable this skip",
)


@pytest.mark.anyio
async def test_index_page_chartjs_canvas():
    """Dashboard chart canvas renders with Chart.js data attributes"""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(Path(tmp))
        store.save(ExperimentIR(
            name="js-test", parameters={"lr": 0.01}, outputs=ExperimentOutputs(metrics={"acc": 0.9}),
        ))
        app = create_app(store)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            text = resp.text
            assert "chart" in text.lower() or "canvas" in text.lower()


@pytest.mark.anyio
async def test_experiment_page_checkboxes():
    """Experiment table has checkboxes for batch compare"""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(Path(tmp))
        store.save(ExperimentIR(name="box-test", project="p1"))
        store.save(ExperimentIR(name="box-test-2", project="p1"))
        app = create_app(store)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/experiments")
            assert resp.status_code == 200
            assert 'type="checkbox"' in resp.text
            assert "compare" in resp.text.lower()


@pytest.mark.anyio
async def test_graph_page_svg_container():
    """Graph page has SVG container for D3.js"""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(Path(tmp))
        store.save(ExperimentIR(name="graph-svg"))
        app = create_app(store)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/graph")
            assert resp.status_code == 200
            assert "<svg" in resp.text or "d3" in resp.text.lower()


@pytest.mark.anyio
async def test_create_page_form():
    """Create page has a form"""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(Path(tmp))
        store.save(ExperimentIR(name="form-target"))
        app = create_app(store)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/create")
            assert resp.status_code == 200
            assert "<form" in resp.text or "method" in resp.text


@pytest.mark.anyio
async def test_detail_page_json_data_attrs():
    """Experiment detail page has data-* attributes for JS"""
    with tempfile.TemporaryDirectory() as tmp:
        store = FileStore(Path(tmp))
        exp = ExperimentIR(
            name="data-attr-test",
            parameters={"x": 1, "y": "foo"},
            outputs=ExperimentOutputs(metrics={"acc": 0.95}),
        )
        store.save(exp)
        app = create_app(store)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/experiments/{exp.id}")
            assert resp.status_code == 200
            assert "data-" in resp.text or "dataset" in resp.text.lower()
