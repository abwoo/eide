from __future__ import annotations

import json
import os
from pathlib import Path

import jinja2
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from eide.diff.engine import DiffEngine
from eide.graph.builder import ExperimentGraph
from eide.intelligence.advisor import ExperimentAdvisor
from eide.storage.filestore import FileStore

HERE = Path(__file__).parent
DEFAULT_ROOT = os.environ.get("EIDE_ROOT", ".eide")


def create_app(store_override: FileStore | None = None) -> FastAPI:
    app = FastAPI(title="EIDE Dashboard")
    _env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(HERE / "templates")),
        autoescape=jinja2.select_autoescape(),
        cache_size=0,
    )
    templates = Jinja2Templates(env=_env)

    def _get_store(root: str | None = None) -> FileStore:
        return store_override or FileStore(root or DEFAULT_ROOT)

    @app.get("/api/experiments")
    async def api_experiments(root: str | None = None, search: str | None = None):
        store = _get_store(root)
        if search:
            exps = store.search(search)
            return [
                {"id": e.id, "name": e.name, "project": e.project, "timestamp": e.timestamp.isoformat(),
                 "tags": e.tags, "pipeline_name": e.pipeline.name}
                for e in exps
            ]
        return store.list_experiments()

    @app.get("/api/experiments/{exp_id}")
    async def api_experiment(exp_id: str, root: str | None = None):
        store = _get_store(root)
        exp = store.load(exp_id)
        return json.loads(exp.model_dump_json())

    @app.post("/api/diff")
    async def api_diff(exp_a: str = Form(...), exp_b: str = Form(...), root: str | None = None):
        store = _get_store(root)
        e_a = store.load(exp_a)
        e_b = store.load(exp_b)
        engine = DiffEngine()
        report = engine.diff(e_a, e_b)
        return json.loads(report.model_dump_json())

    @app.post("/api/explain")
    async def api_explain(exp_a: str = Form(...), exp_b: str = Form(...), root: str | None = None):
        store = _get_store(root)
        e_a = store.load(exp_a)
        e_b = store.load(exp_b)
        advisor = ExperimentAdvisor(store=store)
        insight = advisor.compare(e_a, e_b)
        return insight.to_dict()

    @app.get("/api/graph")
    async def api_graph(root: str | None = None):
        store = _get_store(root)
        eg = ExperimentGraph.build(store)
        nodes = []
        edges = []
        for n, data in eg.graph.nodes(data=True):
            nodes.append({"id": n, "label": data.get("label", n), "type": data.get("type", "unknown")})
        for u, v, data in eg.graph.edges(data=True):
            edges.append({"source": u, "target": v, "type": data.get("type", "unknown")})
        return {"nodes": nodes, "edges": edges}

    @app.get("/api/experiments/{exp_id}/metrics")
    async def api_experiment_metrics(exp_id: str, root: str | None = None):
        """Return metric history as time-series data for charts."""
        store = _get_store(root)
        exp = store.load(exp_id)
        series = {}
        for key, points in exp.outputs.metric_history.items():
            series[key] = [
                {"step": p.step, "value": p.value, "timestamp": p.timestamp.isoformat() if p.timestamp else None}
                for p in points
            ]
        if not series:
            for key, val in exp.outputs.metrics.items():
                series[key] = [{"step": 0, "value": val, "timestamp": None}]
        return series

    @app.get("/api/stats")
    async def api_stats(root: str | None = None):
        store = _get_store(root)
        entries = store.list_experiments()
        if not entries:
            return {"total": 0, "by_tag": {}, "metric_range": {}}

        tag_counts: dict[str, int] = {}
        all_metrics: dict[str, list[float]] = {}
        for e in entries:
            exp = store.load(e["id"])
            for k, v in exp.tags.items():
                tag_counts[f"{k}={v}"] = tag_counts.get(f"{k}={v}", 0) + 1
            for k, v in exp.outputs.metrics.items():
                all_metrics.setdefault(k, []).append(v)

        metric_range = {}
        for k, vals in all_metrics.items():
            metric_range[k] = {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals), "count": len(vals)}

        return {"total": len(entries), "by_tag": tag_counts, "metric_range": metric_range}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, root: str | None = None):
        store = _get_store(root)
        entries = store.list_experiments()
        stats = {}
        if entries:
            all_metrics: dict[str, list[float]] = {}
            tag_counts: dict[str, int] = {}
            for e in entries:
                exp = store.load(e["id"])
                for k, v in exp.tags.items():
                    tag_counts[f"{k}={v}"] = tag_counts.get(f"{k}={v}", 0) + 1
                for k, v in exp.outputs.metrics.items():
                    all_metrics.setdefault(k, []).append(v)
            stats = {
                "total": len(entries),
                "tag_counts": dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:10]),
                "top_metrics": dict(
                    sorted(
                        [(k, round(sum(v) / len(v), 3)) for k, v in all_metrics.items()],
                        key=lambda x: -x[1],
                    )[:8]
                ),
            }
        return templates.TemplateResponse(request, "index.html", {"stats": stats, "entries": entries})

    @app.get("/experiments", response_class=HTMLResponse)
    async def experiments_page(request: Request, root: str | None = None, search: str | None = None):
        store = _get_store(root)
        if search:
            exps = store.search(search)
            entries = [
                {"id": e.id, "name": e.name, "project": e.project, "timestamp": e.timestamp.isoformat(),
                 "tags": e.tags, "pipeline_name": e.pipeline.name}
                for e in exps
            ]
        else:
            entries = store.list_experiments()
        return templates.TemplateResponse(request, "experiments.html", {"entries": entries, "search": search})

    @app.get("/experiments/{exp_id}", response_class=HTMLResponse)
    async def experiment_detail(request: Request, exp_id: str, root: str | None = None):
        store = _get_store(root)
        exp = store.load(exp_id)
        advisor = ExperimentAdvisor(store=store)
        risk = advisor.assess_risk(exp)
        baseline_id = None
        baseline_changes = None
        try:
            rec = advisor.recommend_baseline(exp)
            bm = rec.best_match if rec else None
            if bm:
                baseline_id = bm.get("experiment_id") if isinstance(bm, dict) else bm
                baseline_changes = len(rec.significant_differences or [])
        except Exception:
            pass
        return templates.TemplateResponse(
            request, "experiment_detail.html", {
                "exp": exp, "risk": risk,
                "baseline_id": baseline_id, "baseline_changes": baseline_changes,
            }
        )

    @app.get("/diff", response_class=HTMLResponse)
    async def diff_page(request: Request, a: str | None = None, b: str | None = None, root: str | None = None):
        report = None
        store = _get_store(root)
        entries = store.list_experiments()
        if a and b:
            exp_a = store.load(a)
            exp_b = store.load(b)
            engine = DiffEngine()
            report = engine.diff(exp_a, exp_b)
        return templates.TemplateResponse(
            request,
            "diff.html",
            {
                "entries": entries,
                "report": report,
                "selected_a": a,
                "selected_b": b,
            },
        )

    @app.get("/api/experiments/{exp_id}/risk")
    async def api_experiment_risk(exp_id: str, root: str | None = None):
        store = _get_store(root)
        exp = store.load(exp_id)
        advisor = ExperimentAdvisor(store=store)
        risk = advisor.assess_risk(exp)
        return {"risk_level": risk.risk_level, "risks": risk.risks}

    @app.get("/graph", response_class=HTMLResponse)
    async def graph_page(request: Request, root: str | None = None):
        store = _get_store(root)
        eg = ExperimentGraph.build(store)
        nodes = []
        edges = []
        for n, data in eg.graph.nodes(data=True):
            nodes.append({"id": n, "label": data.get("label", n), "type": data.get("type", "unknown")})
        for u, v, data in eg.graph.edges(data=True):
            edges.append({"source": u, "target": v, "type": data.get("type", "unknown")})
        return templates.TemplateResponse(
            request,
            "graph.html",
            {"nodes": json.dumps(nodes), "edges": json.dumps(edges)},
        )

    @app.get("/replay/{exp_id}", response_class=HTMLResponse)
    async def replay_page(request: Request, exp_id: str, root: str | None = None):
        store = _get_store(root)
        exp = store.load(exp_id)
        return templates.TemplateResponse(request, "replay.html", {"exp": exp})

    return app


app: FastAPI | None = None

def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app


def serve(host: str = "127.0.0.1", port: int = 8765, root: str | None = None):
    import uvicorn

    if root:
        os.environ["EIDE_ROOT"] = root
    print(f"EIDE Dashboard: http://{host}:{port}")
    print(f"Store root: {root or DEFAULT_ROOT}")
    uvicorn.run(get_app(), host=host, port=port)
