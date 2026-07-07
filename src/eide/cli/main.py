from __future__ import annotations

import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from eide.cli.advise import advise as _advise_cmd
from eide.cli.dashboard import dashboard as _dashboard_cmd
from eide.cli.diff import diff as _diff_cmd
from eide.cli.explain import explain as _explain_cmd
from eide.cli.export_import import export as _export_cmd
from eide.cli.export_import import import_exp as _import_cmd
from eide.cli.graph import graph as _graph_cmd
from eide.cli.replay import replay as _replay_cmd
from eide.cli.run import run as _run_cmd
from eide.cli.search import search as _search_cmd
from eide.cli.serve import serve as _serve_cmd
from eide.cli.tag import tag as _tag_cmd
from eide.core.ir import ExperimentIR
from eide.storage.filestore import FileStore

console = Console()

DEFAULT_EIDE_ROOT = os.environ.get("EIDE_ROOT", ".eide")


def _get_store(root: str | None = None) -> FileStore:
    root_path = root or DEFAULT_EIDE_ROOT
    return FileStore(Path(root_path))


@click.group()
def cli():
    """EIDE - Experiment Intelligence & Diff Engine"""


@cli.command()
@click.option("--root", default=DEFAULT_EIDE_ROOT, help="EIDE store root directory")
def init(root: str):
    """Initialize an EIDE experiment store"""
    store = FileStore(root)
    console.print(f"[green]✔ Initialized EIDE store at [bold]{store.root}[/bold][/green]")


@cli.command()
@click.option("--name", default="", help="Experiment name")
@click.option("--project", default="", help="Project/group name")
@click.option("--tag", multiple=True, help="Tags (key=value)")
@click.option("--diff", is_flag=True, help="Auto-diff against most similar baseline")
@click.option("--root", default=None, help="EIDE store root")
def create(name: str, project: str, tag: tuple[str, ...], diff: bool, root: str | None):
    """Create a new experiment"""
    store = _get_store(root)
    exp = ExperimentIR(name=name, project=project)
    for t in tag:
        if "=" in t:
            k, v = t.split("=", 1)
            exp.tags[k] = v
    path = store.save(exp)
    console.print(f"[green]✔ Created experiment [bold]{exp.id}[/bold][/green]")
    console.print(f"  Path: {path}")
    console.print(f"  Name: {name or '(unnamed)'}")

    if diff:
        result = store.diff_with_baseline(exp.id)
        if result:
            console.print(
                f"\n[bold]Auto-diff with baseline [dim]{result['baseline_id'][:12]}[/dim]:[/bold]"
            )
            console.print(f"  [dim]{result['summary']}[/dim]")


@cli.command("list")
@click.option("--tag", multiple=True, help="Filter by tag (key=value)")
@click.option("--project", default=None, help="Filter by project name")
@click.option("--root", default=None, help="EIDE store root")
def list_experiments(tag: tuple[str, ...], project: str | None, root: str | None):
    """List all experiments"""
    store = _get_store(root)
    entries = store.list_experiments()

    if tag:
        filtered = []
        for e in entries:
            matches = True
            for t in tag:
                if "=" in t:
                    k, v = t.split("=", 1)
                    if e.get("tags", {}).get(k) != v:
                        matches = False
                        break
            if matches:
                filtered.append(e)
        entries = filtered

    if project:
        entries = [e for e in entries if e.get("project", "") == project]

    if not entries:
        console.print("[yellow]No experiments found.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Project")
    table.add_column("Timestamp")
    table.add_column("Pipeline")
    table.add_column("Tags")
    for e in entries:
        tags_str = ", ".join(f"{k}={v}" for k, v in e.get("tags", {}).items())
        table.add_row(
            e["id"],
            e.get("name", "") or "(unnamed)",
            e.get("project", "") or "—",
            e.get("timestamp", "")[:19],
            e.get("pipeline_name", ""),
            tags_str,
        )
    console.print(table)


@cli.command()
@click.argument("experiment_id")
@click.option("--root", default=None, help="EIDE store root")
def show(experiment_id: str, root: str | None):
    """Show experiment details"""
    store = _get_store(root)
    try:
        exp = store.load(experiment_id)
    except FileNotFoundError:
        console.print(f"[red]✖ Experiment [bold]{experiment_id}[/bold] not found[/red]")
        return
    console.print(f"[bold cyan]Experiment:[/bold cyan] {exp.id}")
    console.print(f"  Name: {exp.name or '(unnamed)'}")
    console.print(f"  Project: {exp.project or '(none)'}")
    console.print(f"  Timestamp: {exp.timestamp}")
    console.print(f"  Description: {exp.description or '(none)'}")

    if exp.tags:
        console.print("\n[bold]Tags:[/bold]")
        for k, v in exp.tags.items():
            console.print(f"  {k} = {v}")

    if exp.parameters:
        console.print("\n[bold]Parameters:[/bold]")
        for k, v in sorted(exp.parameters.items()):
            console.print(f"  {k} = {v}")

    if exp.pipeline.steps:
        console.print("\n[bold]Pipeline Steps:[/bold]")
        for step in exp.pipeline.steps:
            console.print(f"  {step.name} ({step.type})")

    if exp.outputs.metrics:
        console.print("\n[bold]Metrics:[/bold]")
        for k, v in sorted(exp.outputs.metrics.items()):  # type: ignore[assignment]
            console.print(f"  {k} = {v}")

    if exp.outputs.figures:
        console.print("\n[bold]Figures:[/bold]")
        for f in exp.outputs.figures:
            console.print(f"  {f}")

    if exp.outputs.artifacts:
        console.print("\n[bold]Artifacts:[/bold]")
        for a in exp.outputs.artifacts:
            console.print(f"  {a}")

    if exp.data_versions:
        console.print("\n[bold]Data Versions:[/bold]")
        for k, v in sorted(exp.data_versions.items()):
            console.print(f"  {k} = {v}")

    if exp.decisions:
        console.print("\n[bold]Decisions:[/bold]")
        for k, v in sorted(exp.decisions.items()):
            console.print(f"  {k} = {v}")

    if exp.metadata:
        console.print("\n[bold]Metadata:[/bold]")
        for k, v in sorted(exp.metadata.items()):
            console.print(f"  {k} = {v}")

    if exp.execution_trace:
        console.print("\n[bold]Execution Trace:[/bold]")
        for evt in exp.execution_trace:
            ts = evt.timestamp.isoformat()[:19] if evt.timestamp else "?"
            console.print(f"  {ts} | {evt.step} | {evt.action}")
            if evt.details:
                for dk, dv in sorted(evt.details.items()):
                    console.print(f"    {dk}: {dv}")

    console.print(
        f"\n[dim]Environment: {exp.environment.os} | Python {exp.environment.python_version}[/dim]"
    )


@cli.command()
@click.argument("experiment_id")
@click.option("--root", default=None, help="EIDE store root")
def delete(experiment_id: str, root: str | None):
    """Delete an experiment"""
    store = _get_store(root)
    try:
        store.delete(experiment_id)
    except FileNotFoundError as e:
        console.print(f"[red]✖ {e}[/red]")
        return
    console.print(f"[green]✔ Deleted experiment [bold]{experiment_id}[/bold][/green]")


@cli.group()
def capture():
    """Capture experiments from various sources"""


@capture.command()
@click.option("--name", default="", help="Experiment name")
@click.option("--description", default="", help="Experiment description")
@click.option("--project", default="", help="Project/group name")
@click.option("--param", multiple=True, help="Parameters (key=value)")
@click.option("--metric", multiple=True, help="Metrics (key=value)")
@click.option("--tag", multiple=True, help="Tags (key=value)")
@click.option("--step", multiple=True, help="Pipeline steps (name:type)")
@click.option("--data-version", multiple=True, help="Data versions (key=value)")
@click.option("--decision", multiple=True, help="Decisions (key=value)")
@click.option("--metadata", multiple=True, help="Metadata (key=value)")
@click.option("--diff", is_flag=True, help="Auto-diff against most similar baseline")
@click.option("--root", default=None, help="EIDE store root")
def manual(
    name: str,
    description: str,
    project: str,
    param: tuple[str, ...],
    metric: tuple[str, ...],
    tag: tuple[str, ...],
    step: tuple[str, ...],
    data_version: tuple[str, ...],
    decision: tuple[str, ...],
    metadata: tuple[str, ...],
    diff: bool,
    root: str | None,
):
    """Manually capture an experiment"""
    from eide.capture.manual import capture_manual

    def _parse_kv(items: tuple[str, ...]) -> dict[str, str]:
        result = {}
        for item in items:
            if "=" in item:
                k, v = item.split("=", 1)
                result[k] = v
        return result

    steps_list = []
    for s in step:
        if ":" in s:
            name_part, type_part = s.split(":", 1)
            steps_list.append({"name": name_part, "type": type_part})
        else:
            steps_list.append({"name": s, "type": ""})

    params = _parse_kv(param)
    metrics_dict = {k: float(v) for k, v in _parse_kv(metric).items()}
    tags_dict = _parse_kv(tag)
    data_versions_dict = _parse_kv(data_version)
    decisions_dict = _parse_kv(decision)
    metadata_dict = _parse_kv(metadata)

    exp = capture_manual(
        name=name,
        description=description,
        project=project,
        parameters=params,
        metrics=metrics_dict,
        tags=tags_dict,
        pipeline_steps=steps_list if steps_list else None,
        data_versions=data_versions_dict or None,
        decisions=decisions_dict or None,
        metadata=metadata_dict or None,
    )
    store = _get_store(root)
    store.save(exp)
    console.print(f"[green]✔ Captured experiment [bold]{exp.id}[/bold][/green]")

    if diff:
        result = store.diff_with_baseline(exp.id)
        if result:
            console.print(
                f"\n[bold]Auto-diff with baseline [dim]{result['baseline_id'][:12]}[/dim]:[/bold]"
            )
            console.print(f"  [dim]{result['summary']}[/dim]")
            console.print(f"  [dim]{result['change_count']} change(s)[/dim]")
        else:
            console.print("[dim]No baseline found for comparison[/dim]")


@capture.command()
@click.argument("run_id")
@click.option("--name", default=None, help="Override experiment name")
@click.option("--project", default=None, help="Override project name")
@click.option("--root", default=None, help="EIDE store root")
def mlflow(run_id: str, name: str | None, project: str | None, root: str | None):
    """Capture an MLflow run"""
    from eide.capture.mlflow_adapter import from_mlflow

    try:
        exp = from_mlflow(run_id)
    except ImportError:
        console.print("[red]MLflow not installed. Run: pip install eide-core[mlflow][/red]")
        return

    if name:
        exp.name = name
    if project:
        exp.project = project

    store = _get_store(root)
    store.save(exp)
    console.print(
        (
            f"[green]✔ Captured MLflow run [bold]{run_id}[/bold]"
            f" as experiment [bold]{exp.id}[/bold][/green]"
        )
    )


cli.add_command(_export_cmd)
cli.add_command(_import_cmd)
cli.add_command(_tag_cmd)
cli.add_command(_search_cmd)
cli.add_command(_run_cmd)
cli.add_command(_diff_cmd)
cli.add_command(_serve_cmd)
cli.add_command(_explain_cmd)
cli.add_command(_replay_cmd)
cli.add_command(_graph_cmd)
cli.add_command(_advise_cmd)
cli.add_command(_dashboard_cmd)
