from __future__ import annotations

import json
from pathlib import Path

from eide.core.ir import ExperimentIR
from eide.core.types import (
    ExperimentOutputs,
    PipelineSpec,
    PipelineStep,
)
from eide.diff.engine import DiffEngine
from eide.storage.filestore import FileStore

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore


def _get_store(eide_root: str | None = None) -> FileStore:
    root = Path(eide_root or ".eide")
    return FileStore(root)


async def _do_capture(
    store: FileStore,
    name: str = "",
    description: str = "",
    parameters: str = "{}",
    metrics: str = "{}",
    pipeline_steps: str = "[]",
    tags: str = "{}",
) -> str:
    exp = ExperimentIR(
        name=name,
        description=description,
        parameters=json.loads(parameters),
        outputs=ExperimentOutputs(metrics={k: float(v) for k, v in json.loads(metrics).items()}),
        tags=json.loads(tags),
    )
    steps_data = json.loads(pipeline_steps)
    if steps_data:
        exp.pipeline = PipelineSpec(
            name=name,
            steps=[
                PipelineStep(**s) if isinstance(s, dict) else PipelineStep(name=s)
                for s in steps_data
            ],
        )
    path = store.save(exp)
    return json.dumps({"id": exp.id, "path": str(path)}, indent=2)


async def _do_diff(
    store: FileStore,
    engine: DiffEngine,
    experiment_a: str,
    experiment_b: str,
    artifacts_root: str | None = None,
) -> str:
    try:
        exp_a = store.load(experiment_a)
        exp_b = store.load(experiment_b)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    if artifacts_root:
        engine.artifacts_root = Path(artifacts_root)
    report = engine.diff(exp_a, exp_b)
    return report.model_dump_json(indent=2)


async def _do_search(store: FileStore, query: str, field: str = "all") -> str:
    results = store.search(query, field=field)
    return json.dumps([r.model_dump(mode="json") for r in results], indent=2, default=str)


async def _do_list(store: FileStore) -> str:
    entries = store.list_experiments()
    return json.dumps(entries, indent=2, default=str)


async def _do_show(store: FileStore, experiment_id: str) -> str:
    try:
        exp = store.load(experiment_id)
        return exp.model_dump_json(indent=2)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})


async def _do_delete(store: FileStore, experiment_id: str) -> str:
    try:
        store.delete(experiment_id)
        return json.dumps({"status": "deleted", "id": experiment_id})
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})


async def _do_explain(
    store: FileStore, engine: DiffEngine, experiment_a: str, experiment_b: str
) -> str:
    try:
        exp_a = store.load(experiment_a)
        exp_b = store.load(experiment_b)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    report = engine.diff(exp_a, exp_b)
    result = {
        "experiment_a": experiment_a,
        "experiment_b": experiment_b,
        "summary": report.summary,
        "changelog": report.changelog,
        "impact_estimates": [ie.model_dump() for ie in report.impact_estimates],
        "change_count": len(report.changes),
    }
    param_changes = [c for c in report.changes if c.category.value == "parameter"]
    metric_changes = [c for c in report.changes if c.category.value == "output"]
    env_changes = [c for c in report.changes if c.category.value == "environment"]
    if metric_changes and param_changes:
        top_param = max(param_changes, key=lambda c: c.significance)
        top_metric = max(metric_changes, key=lambda c: c.significance)
        old_v, new_v = top_metric.old_value, top_metric.new_value
        if isinstance(old_v, (int, float)) and isinstance(new_v, (int, float)) and old_v:
            change_pct = ((new_v - old_v) / abs(old_v)) * 100
        else:
            change_pct = 0.0
        result["likely_cause"] = (
            f"Parameter change '{top_param.key}' ({top_param.significance:.0%} significance) "
            f"likely caused metric change '{top_metric.key}' ({change_pct:.1f}% change)"
        )
    elif param_changes:
        result["likely_cause"] = (
            f"Parameter changes detected ({len(param_changes)}), but no metric changes to correlate"
        )
    elif env_changes:
        result["likely_cause"] = f"Environment changed ({len(env_changes)} differences)"
    else:
        result["likely_cause"] = "No significant changes detected"
    return json.dumps(result, indent=2, default=str)


async def _do_graph_duplicates(store: FileStore, experiment_id: str) -> str:
    try:
        from eide.graph.builder import ExperimentGraph
        from eide.graph.queries import GraphQuery

        eg = ExperimentGraph.build(store)
        q = GraphQuery(eg)
        results = q.duplicates(experiment_id)
        return json.dumps(results, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _do_graph_conflicts(store: FileStore, experiment_id: str) -> str:
    try:
        from eide.graph.builder import ExperimentGraph
        from eide.graph.queries import GraphQuery

        eg = ExperimentGraph.build(store)
        q = GraphQuery(eg)
        results = q.conflicts(experiment_id)
        return json.dumps(results, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _do_graph_lineage(store: FileStore, experiment_id: str) -> str:
    try:
        from eide.graph.builder import ExperimentGraph
        from eide.graph.queries import GraphQuery

        eg = ExperimentGraph.build(store)
        q = GraphQuery(eg)
        results = q.lineage(experiment_id)
        return json.dumps(results, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _do_advise_risk(store: FileStore, experiment_id: str) -> str:
    try:
        from eide.intelligence.advisor import ExperimentAdvisor

        exp = store.load(experiment_id)
        advisor = ExperimentAdvisor(store=store)
        assessment = advisor.assess_risk(exp)
        return json.dumps(
            {
                "experiment_id": assessment.experiment_id,
                "risk_level": assessment.risk_level,
                "risks": assessment.risks,
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _do_advise_baseline(store: FileStore, experiment_id: str) -> str:
    try:
        from eide.intelligence.advisor import ExperimentAdvisor

        exp = store.load(experiment_id)
        advisor = ExperimentAdvisor(store=store)
        rec = advisor.recommend_baseline(exp)
        return json.dumps(
            {
                "target_id": rec.target_id,
                "best_match": rec.best_match,
                "significant_differences": [
                    {"key": d.key, "description": d.description}
                    for d in (rec.significant_differences or [])
                ],
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _do_replay(
    store: FileStore,
    experiment_id: str,
    restore_environment: bool = False,
    restore_data: bool = False,
) -> str:
    try:
        from eide.replay.engine import ReplayEngine

        exp = store.load(experiment_id)
        engine = ReplayEngine(store=store)
        result = engine.replay(
            exp,
            restore_environment=restore_environment,
            restore_data=restore_data,
            save_replay=True,
        )
        engine.cleanup()
        return json.dumps(result.to_dict(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _do_graph_param_impact(store: FileStore, param_key: str) -> str:
    try:
        from eide.graph.builder import ExperimentGraph
        from eide.graph.queries import GraphQuery

        eg = ExperimentGraph.build(store)
        q = GraphQuery(eg)
        results = q.parameter_impact(param_key)
        return json.dumps(results, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _do_graph_top_params(store: FileStore, top_n: int = 10) -> str:
    try:
        from eide.graph.builder import ExperimentGraph
        from eide.graph.queries import GraphQuery

        eg = ExperimentGraph.build(store)
        q = GraphQuery(eg)
        results = q.most_used_parameters(top_n=top_n)
        return json.dumps(results, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _create_mcp_server(eide_root: str | None = None) -> "FastMCP":
    if FastMCP is None:
        raise ImportError("mcp package is required. Install: pip install mcp")

    mcp = FastMCP(
        "eide",
        instructions="Experiment Intelligence & Diff Engine - understand why experiments differ",
    )

    store = _get_store(eide_root)
    engine = DiffEngine()

    @mcp.tool()
    async def eide_capture(
        name: str = "",
        description: str = "",
        parameters: str = "{}",
        metrics: str = "{}",
        pipeline_steps: str = "[]",
        tags: str = "{}",
    ) -> str:
        return await _do_capture(
            store, name, description, parameters, metrics, pipeline_steps, tags
        )

    @mcp.tool()
    async def eide_diff(
        experiment_a: str,
        experiment_b: str,
        artifacts_root: str | None = None,
    ) -> str:
        return await _do_diff(store, engine, experiment_a, experiment_b, artifacts_root)

    @mcp.tool()
    async def eide_search(query: str, field: str = "all") -> str:
        return await _do_search(store, query, field)

    @mcp.tool()
    async def eide_list() -> str:
        return await _do_list(store)

    @mcp.tool()
    async def eide_show(experiment_id: str) -> str:
        return await _do_show(store, experiment_id)

    @mcp.tool()
    async def eide_delete(experiment_id: str) -> str:
        return await _do_delete(store, experiment_id)

    @mcp.tool()
    async def eide_explain(experiment_a: str, experiment_b: str) -> str:
        return await _do_explain(store, engine, experiment_a, experiment_b)

    @mcp.resource("eide://experiments")
    async def experiments_index() -> str:
        return await _do_list(store)

    @mcp.resource("eide://experiments/{experiment_id}")
    async def experiment_by_id(experiment_id: str) -> str:
        return await _do_show(store, experiment_id)

    @mcp.tool()
    async def eide_graph_duplicates(experiment_id: str) -> str:
        return await _do_graph_duplicates(store, experiment_id)

    @mcp.tool()
    async def eide_graph_conflicts(experiment_id: str) -> str:
        return await _do_graph_conflicts(store, experiment_id)

    @mcp.tool()
    async def eide_graph_lineage(experiment_id: str) -> str:
        return await _do_graph_lineage(store, experiment_id)

    @mcp.tool()
    async def eide_advise_risk(experiment_id: str) -> str:
        return await _do_advise_risk(store, experiment_id)

    @mcp.tool()
    async def eide_advise_baseline(experiment_id: str) -> str:
        return await _do_advise_baseline(store, experiment_id)

    @mcp.tool()
    async def eide_replay(
        experiment_id: str,
        restore_environment: bool = False,
        restore_data: bool = False,
    ) -> str:
        """Replay an experiment and verify reproducibility.
        Set restore_environment=True and restore_data=True for a full
        reproducibility check (creates venv, restores data files)."""
        return await _do_replay(
            store,
            experiment_id,
            restore_environment,
            restore_data,
        )

    @mcp.tool()
    async def eide_graph_param_impact(param_key: str) -> str:
        """Show how different values of a parameter correlate with metrics"""
        return await _do_graph_param_impact(store, param_key)

    @mcp.tool()
    async def eide_graph_top_params(top_n: int = 10) -> str:
        return await _do_graph_top_params(store, top_n)

    return mcp


def serve(eide_root: str | None = None, host: str = "127.0.0.1", port: int = 8080):
    """Start the MCP server (SSE transport)."""
    try:
        mcp = _create_mcp_server(eide_root)
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport="sse")
    except ImportError as e:
        print(f"Error: {e}")
        print("Install mcp and uvicorn: pip install mcp uvicorn")
        raise
