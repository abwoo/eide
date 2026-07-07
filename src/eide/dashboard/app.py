from __future__ import annotations

import json
import os
import time
from pathlib import Path, PurePath
from typing import Any

import jinja2
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from eide.diff.engine import DiffEngine
from eide.graph.builder import ExperimentGraph
from eide.intelligence.advisor import ExperimentAdvisor
from eide.storage.filestore import FileStore

try:
    from importlib.metadata import version as _version

    EIDE_VERSION = _version("eide-core")
except Exception:
    EIDE_VERSION = "1.0.0"

HERE = Path(__file__).parent
DEFAULT_ROOT = os.environ.get("EIDE_ROOT", ".eide")


def _resolve_root(root: str | None = None) -> Path:
    if root is None:
        return Path(DEFAULT_ROOT)
    parts = PurePath(root).parts
    if ".." in parts:
        raise HTTPException(status_code=400, detail="Invalid root path")
    candidate = Path(root).resolve()
    return candidate


def _common_context() -> dict:
    return {"version": EIDE_VERSION}


def create_app(store_override: FileStore | None = None) -> FastAPI:
    app = FastAPI(title="EIDE Dashboard")
    _env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(HERE / "templates")),
        autoescape=jinja2.select_autoescape(),
        cache_size=0,
    )
    templates = Jinja2Templates(env=_env)

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        start_t = time.time()
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        elapsed = time.time() - start_t
        response.headers["X-Response-Time"] = f"{elapsed * 1000:.0f}ms"
        return response

    def _get_store() -> FileStore:
        if store_override:
            return store_override
        return FileStore(_resolve_root(None))

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": str(exc)}, status_code=404)
        return templates.TemplateResponse(
            request,
            "error.html",
            {"code": 404, "message": str(exc), **_common_context()},
            status_code=404,
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": str(exc)}, status_code=500)
        return templates.TemplateResponse(
            request,
            "error.html",
            {"code": 500, "message": str(exc), **_common_context()},
            status_code=500,
        )

    def _experiment_entry(exp) -> dict:
        return {
            "id": exp.id,
            "name": exp.name,
            "project": exp.project or "",
            "timestamp": exp.timestamp.isoformat() if hasattr(exp, "timestamp") else "",
            "tags": getattr(exp, "tags", {}),
            "pipeline_name": getattr(exp.pipeline, "name", "") if hasattr(exp, "pipeline") else "",
            "metrics": dict(getattr(exp.outputs, "metrics", {})),
            "parameters": dict(getattr(exp, "parameters", {})),
        }

    def _load_entries(store: FileStore, limit: int = 0) -> list[dict]:
        entries = store.list_experiments()
        if limit:
            entries = entries[:limit]
        out = []
        for e in entries:
            try:
                exp = store.load(e["id"])
                out.append(_experiment_entry(exp))
            except Exception:
                out.append(e)
        return out

    def _group_by_project(entries: list[dict]) -> list[dict]:
        project_map: dict[str, list[dict]] = {}
        for e in entries:
            p = e.get("project") or ""
            project_map.setdefault(p, []).append(e)
        projects = []
        for pname, exps in sorted(project_map.items(), key=lambda x: -len(x[1])):
            projects.append(
                {
                    "name": pname or "(default)",
                    "count": len(exps),
                    "experiments": sorted(exps, key=lambda x: x.get("timestamp", ""), reverse=True)[
                        :5
                    ],
                }
            )
        return projects

    @app.get("/api/experiments")
    def api_experiments(search: str | None = None, limit: int = 0, offset: int = 0):
        store = _get_store()
        if search:
            exps = store.search(search)
            results = [_experiment_entry(e) for e in exps]
        else:
            results = _load_entries(store)
        total = len(results)
        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]
        return {"total": total, "offset": offset, "limit": limit, "items": results}

    @app.get("/api/projects")
    def api_projects():
        store = _get_store()
        entries = _load_entries(store)
        return {"projects": _group_by_project(entries)}

    @app.get("/api/experiments/{exp_id}")
    def api_experiment(exp_id: str):
        store = _get_store()
        exp = store.load(exp_id)
        return json.loads(exp.model_dump_json())

    @app.post("/api/diff")
    def api_diff(exp_a: str = Form(...), exp_b: str = Form(...)):
        store = _get_store()
        e_a = store.load(exp_a)
        e_b = store.load(exp_b)
        engine = DiffEngine()
        report = engine.diff(e_a, e_b)
        return json.loads(report.model_dump_json())

    @app.post("/api/explain")
    def api_explain(exp_a: str = Form(...), exp_b: str = Form(...)):
        store = _get_store()
        e_a = store.load(exp_a)
        e_b = store.load(exp_b)
        advisor = ExperimentAdvisor(store=store)
        insight = advisor.compare(e_a, e_b)
        return insight.to_dict()

    @app.get("/api/graph")
    def api_graph():
        store = _get_store()
        eg = ExperimentGraph.build(store)
        nodes = []
        edges = []
        for n, data in eg.graph.nodes(data=True):
            nodes.append(
                {"id": n, "label": data.get("label", n), "type": data.get("type", "unknown")}
            )
        for u, v, data in eg.graph.edges(data=True):
            edges.append({"source": u, "target": v, "type": data.get("type", "unknown")})
        return {"nodes": nodes, "edges": edges}

    @app.get("/api/experiments/{exp_id}/metrics")
    def api_experiment_metrics(exp_id: str):
        store = _get_store()
        exp = store.load(exp_id)
        series = {}
        for key, points in exp.outputs.metric_history.items():
            series[key] = [
                {
                    "step": p.step,
                    "value": p.value,
                    "timestamp": p.timestamp.isoformat() if p.timestamp else None,
                }
                for p in points
            ]
        if not series:
            for key, val in exp.outputs.metrics.items():
                series[key] = [{"step": 0, "value": val, "timestamp": None}]
        return series

    @app.get("/api/experiments/{exp_id}/risk")
    def api_experiment_risk(exp_id: str):
        store = _get_store()
        exp = store.load(exp_id)
        advisor = ExperimentAdvisor(store=store)
        risk = advisor.assess_risk(exp)
        return {"risk_level": risk.risk_level, "risks": risk.risks}

    @app.get("/api/stats")
    def api_stats():
        store = _get_store()
        entries = _load_entries(store)
        if not entries:
            return {"total": 0, "projects": 0, "by_tag": {}, "metric_range": {}}
        tag_counts: dict[str, int] = {}
        all_metrics: dict[str, list[float]] = {}
        projects: set[str] = set()
        total_params = 0
        for e in entries:
            projects.add(e.get("project", "") or "")
            for k, v in e.get("tags", {}).items():
                tag_counts[f"{k}={v}"] = tag_counts.get(f"{k}={v}", 0) + 1
            for k, v in e.get("metrics", {}).items():
                try:
                    all_metrics.setdefault(k, []).append(float(v))
                except (ValueError, TypeError):
                    pass
            total_params += len(e.get("parameters", {}))
        metric_range = {}
        for k, vals in all_metrics.items():
            metric_range[k] = {
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
                "count": len(vals),
            }
        return {
            "total": len(entries),
            "projects": len(projects),
            "total_tags": len(tag_counts),
            "total_params": total_params,
            "by_tag": tag_counts,
            "metric_range": metric_range,
        }

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        store = _get_store()
        entries = _load_entries(store, limit=200)
        stats = {}
        if entries:
            all_metrics: dict[str, list[float]] = {}
            tag_counts: dict[str, int] = {}
            projects: set[str] = set()
            for e in entries:
                projects.add(e.get("project", "") or "")
                for k, v in e.get("tags", {}).items():
                    tag_counts[f"{k}={v}"] = tag_counts.get(f"{k}={v}", 0) + 1
                for k, v in e.get("metrics", {}).items():
                    try:
                        all_metrics.setdefault(k, []).append(float(v))
                    except (ValueError, TypeError):
                        pass
            stats = {
                "total": len(entries),
                "projects": len(projects),
                "tag_counts": dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:10]),
                "top_metrics": dict(
                    sorted(
                        [(k, round(sum(v) / len(v), 4)) for k, v in all_metrics.items()],
                        key=lambda x: -abs(x[1]),
                    )[:8]
                ),
            }
        project_groups = _group_by_project(entries)
        recent = sorted(entries, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "stats": stats,
                "entries": entries,
                "project_groups": project_groups,
                "recent": recent,
                **_common_context(),
            },
        )

    @app.get("/experiments", response_class=HTMLResponse)
    def experiments_page(
        request: Request,
        search: str | None = None,
        project: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ):
        store = _get_store()
        if search:
            exps = store.search(search)
            entries = [
                _experiment_entry(e) for e in exps
            ]
        elif project:
            index = store.list_experiments()
            matching_ids = [e["id"] for e in index if (e.get("project") or "") == project]
            entries = []
            for eid in matching_ids:
                try:
                    exp = store.load(eid)
                    entries.append(_experiment_entry(exp))
                except Exception:
                    pass
            project_list = [project]
        else:
            entries = _load_entries(store)
            project_set = {e.get("project", "") or "(default)" for e in entries}
            project_list = sorted(project_set)

        if project and not search:
            total = len(entries)
        else:
            total = len(entries)
            if project:
                entries = [e for e in entries if (e.get("project") or "") == project]
                project_set = {e.get("project", "") or "(default)" for e in entries}
                project_list = sorted(project_set)

        page_entries = entries[offset : offset + limit]
        page_num = offset // limit + 1 if limit else 1
        total_pages = (total + limit - 1) // limit if limit else 1
        return templates.TemplateResponse(
            request,
            "experiments.html",
            {
                "entries": page_entries,
                "search": search,
                "current_project": project,
                "project_list": project_list,
                "page_num": page_num,
                "total_pages": total_pages,
                "total": total,
                "limit": limit,
                "offset": offset,
                **_common_context(),
            },
        )

    @app.get("/experiments/{exp_id}", response_class=HTMLResponse)
    def experiment_detail(request: Request, exp_id: str):
        store = _get_store()
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
            request,
            "experiment_detail.html",
            {
                "exp": exp,
                "risk": risk,
                "baseline_id": baseline_id,
                "baseline_changes": baseline_changes,
                **_common_context(),
            },
        )

    @app.get("/diff", response_class=HTMLResponse)
    def diff_page(request: Request, a: str | None = None, b: str | None = None):
        report = None
        store = _get_store()
        entries = _load_entries(store)
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
                **_common_context(),
            },
        )

    @app.get("/graph", response_class=HTMLResponse)
    def graph_page(request: Request):
        store = _get_store()
        eg = ExperimentGraph.build(store)
        nodes = []
        edges = []
        for n, data in eg.graph.nodes(data=True):
            nodes.append(
                {"id": n, "label": data.get("label", n), "type": data.get("type", "unknown")}
            )
        for u, v, data in eg.graph.edges(data=True):
            edges.append({"source": u, "target": v, "type": data.get("type", "unknown")})
        return templates.TemplateResponse(
            request,
            "graph.html",
            {
                "nodes_json": json.dumps(nodes),
                "edges_json": json.dumps(edges),
                "node_count": len(nodes),
                "edge_count": len(edges),
                **_common_context(),
            },
        )

    @app.get("/replay/{exp_id}", response_class=HTMLResponse)
    def replay_page(request: Request, exp_id: str):
        store = _get_store()
        exp = store.load(exp_id)
        return templates.TemplateResponse(request, "replay.html", {"exp": exp, **_common_context()})

    @app.get("/create", response_class=HTMLResponse)
    def create_page(request: Request):
        store = _get_store()
        projects = sorted(set(e.get("project", "") or "(default)" for e in _load_entries(store)))
        return templates.TemplateResponse(
            request, "create.html", {"projects": projects, **_common_context()}
        )

    @app.post("/api/create")
    def api_create(
        name: str = Form(""),
        project: str = Form(""),
        description: str = Form(""),
        tags: str = Form(""),
        params: str = Form(""),
        metrics: str = Form(""),
    ):
        store = _get_store()
        parsed_params = {}
        for p in params.split(",") if params else []:
            if "=" in p:
                k, v = p.split("=", 1)
                parsed_params[k.strip()] = v.strip()
        parsed_metrics: dict[str, Any] = {}
        for m in metrics.split(",") if metrics else []:
            if "=" in m:
                k, v = m.split("=", 1)
                try:
                    parsed_metrics[k.strip()] = float(v.strip())
                except ValueError:
                    parsed_metrics[k.strip()] = v.strip()
        parsed_tags = {}
        for t in tags.split(",") if tags else []:
            if "=" in t:
                k, v = t.split("=", 1)
                parsed_tags[k.strip()] = v.strip()
        from eide.capture.manual import capture_manual

        exp = capture_manual(
            name=name,
            description=description,
            project=project,
            parameters=parsed_params,
            metrics=parsed_metrics,
            tags=parsed_tags,
        )
        store.save(exp)
        return {"id": exp.id, "name": exp.name, "status": "created"}

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
