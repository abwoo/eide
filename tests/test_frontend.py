from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import expect

from eide.core.ir import ExperimentIR
from eide.core.types import ExperimentOutputs
from eide.dashboard.app import create_app
from eide.storage.filestore import FileStore


@pytest.fixture(scope="module")
def _server():
    """Start the dashboard app on a real port for browser testing"""
    store = FileStore(Path(tempfile.mkdtemp()))
    store.save(ExperimentIR(
        name="test-run", project="demo",
        parameters={"lr": 0.01, "epochs": 5},
        outputs=ExperimentOutputs(metrics={"acc": 0.95, "loss": 0.12}),
    ))
    store.save(ExperimentIR(
        name="test-run-2", project="demo",
        parameters={"lr": 0.02, "epochs": 10},
        outputs=ExperimentOutputs(metrics={"acc": 0.97, "loss": 0.08}),
    ))
    store.save(ExperimentIR(
        name="standalone", project="other",
        parameters={"batch": 32},
        outputs=ExperimentOutputs(metrics={"f1": 0.88}),
    ))

    app = create_app(store)
    host = "127.0.0.1"
    port = 9876
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import time
    time.sleep(1)
    yield f"http://{host}:{port}"
    server.should_exit = True
    thread.join(timeout=5)


class TestFrontend:
    def test_index_page_shows_stats(self, page, _server):
        page.goto(f"{_server}/")
        expect(page.locator("body")).to_contain_text("Experiments")
        expect(page.locator("body")).to_contain_text("Projects")

    def test_experiments_page_checkboxes(self, page, _server):
        page.goto(f"{_server}/experiments")
        checkboxes = page.locator('input[type="checkbox"]')
        expect(checkboxes.first).to_be_visible()

    def test_experiment_detail_shows_params(self, page, _server):
        page.goto(f"{_server}/experiments")
        link = page.locator("a:has-text('test-run')").first
        link.click()
        page.wait_for_url("**/experiments/**")
        expect(page.locator("body")).to_contain_text("0.01")
        expect(page.locator("body")).to_contain_text("lr")

    def test_diff_page_selects(self, page, _server):
        page.goto(f"{_server}/diff")
        select_a = page.locator('select[name="a"], select[id*="a"]').first
        select_b = page.locator('select[name="b"], select[id*="b"]').first
        expect(select_a).to_be_visible()
        expect(select_b).to_be_visible()

    def test_create_page_has_form(self, page, _server):
        page.goto(f"{_server}/create")
        expect(page.locator("#create-form")).to_be_visible()

    def test_graph_page_has_svg_or_canvas(self, page, _server):
        page.goto(f"{_server}/graph")
        svg = page.locator("svg")
        canvas = page.locator("canvas")
        assert svg.count() > 0 or canvas.count() > 0

    def test_replay_page_shows_info(self, page, _server):
        page.goto(f"{_server}/experiments")
        link = page.locator("a:has-text('test-run')").first
        link.click()
        page.wait_for_url("**/experiments/**")
        replay_link = page.locator('a[href*="replay"]').first
        if replay_link.count() > 0:
            replay_link.click()
            page.wait_for_url("**/replay/**")
            expect(page.locator("body")).to_contain_text("test-run")
