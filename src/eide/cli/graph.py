from __future__ import annotations

import os

import click
from rich.console import Console
from rich.table import Table

from eide.graph.builder import ExperimentGraph
from eide.graph.queries import GraphQuery
from eide.storage.filestore import FileStore

console = Console()
DEFAULT_EIDE_ROOT = os.environ.get("EIDE_ROOT", ".eide")


@click.group()
def graph():
    """Experiment knowledge graph queries"""


@graph.command()
@click.option("--root", default=None, help="EIDE store root")
@click.option("--max", "max_exp", default=None, type=int, help="Max experiments to include")
@click.option("--save", default=None, help="Save graph to file")
def build(root: str | None, max_exp: int | None, save: str | None):
    """Build the experiment knowledge graph"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    with console.status("[bold green]Building knowledge graph..."):
        eg = ExperimentGraph.build(store, max_experiments=max_exp)
    node_count = eg.graph.number_of_nodes()
    edge_count = eg.graph.number_of_edges()
    console.print(f"[green]✔ Built graph: {node_count} nodes, {edge_count} edges[/green]")

    if save:
        eg.save(save)
        console.print(f"[green]✔ Saved to {save}[/green]")

    type_counts: dict[str, int] = {}
    for _, data in eg.graph.nodes(data=True):
        t = data.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    console.print("\n[bold]Node types:[/bold]")
    for t, c in sorted(type_counts.items()):
        console.print(f"  {t}: {c}")


@graph.command()
@click.argument("experiment_id")
@click.option("--root", default=None, help="EIDE store root")
@click.option("--save", default=None, help="Graph file path (if already built)")
def duplicates(experiment_id: str, root: str | None, save: str | None):
    """Find duplicate experiments"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    eg = _load_or_build(store, save)
    q = GraphQuery(eg)
    results = q.duplicates(experiment_id)

    if not results:
        console.print(f"[yellow]No similar experiments found for {experiment_id}[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Experiment ID")
    table.add_column("Label")
    table.add_column("Similarity")
    for r in results:
        table.add_row(r["id"], r.get("label", ""), f"{r['similarity']:.1%}")
    console.print(table)


@graph.command()
@click.argument("experiment_id")
@click.option("--root", default=None, help="EIDE store root")
@click.option("--save", default=None, help="Graph file path")
def conflicts(experiment_id: str, root: str | None, save: str | None):
    """Find experiments with conflicting results"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    eg = _load_or_build(store, save)
    q = GraphQuery(eg)
    results = q.conflicts(experiment_id)

    if not results:
        console.print(f"[green]No conflicting experiments found for {experiment_id}[/green]")
        return
    for r in results:
        console.print(f"  • {r['id']} ({r.get('label', '')})")


@graph.command()
@click.argument("experiment_id")
@click.option("--root", default=None, help="EIDE store root")
@click.option("--save", default=None, help="Graph file path")
def lineage(experiment_id: str, root: str | None, save: str | None):
    """Show experiment replay lineage"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    eg = _load_or_build(store, save)
    q = GraphQuery(eg)
    chain = q.lineage(experiment_id)

    if not chain:
        console.print(f"[yellow]No lineage found for {experiment_id}[/yellow]")
        return
    console.print(f"[bold]Lineage for {experiment_id}:[/bold]")
    for i, item in enumerate(chain):
        prefix = " └─ " if i == len(chain) - 1 else " ├─ "
        console.print(f"  {prefix}[cyan]{item['id']}[/cyan] ({item.get('label', '')})")


@graph.command()
@click.option("--root", default=None, help="EIDE store root")
@click.option("--top", default=10, type=int, help="Number of top parameters")
@click.option("--save", default=None, help="Graph file path")
def top_params(root: str | None, top: int, save: str | None):
    """Show most commonly used parameters"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    eg = _load_or_build(store, save)
    q = GraphQuery(eg)
    results = q.most_used_parameters(top_n=top)

    if not results:
        console.print("[yellow]No parameters found[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column("Value")
    table.add_column("Count")
    for r in results:
        table.add_row(r["key"], str(r["value"]), str(r["count"]))
    console.print(table)


@graph.command()
@click.argument("param_key")
@click.option("--root", default=None, help="EIDE store root")
@click.option("--save", default=None, help="Graph file path")
def param_impact(param_key: str, root: str | None, save: str | None):
    """Show how a parameter value correlates with metrics"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    eg = _load_or_build(store, save)
    q = GraphQuery(eg)
    results = q.parameter_impact(param_key)

    if not results:
        console.print(f"[yellow]No data for parameter '{param_key}'[/yellow]")
        return
    for r in results:
        console.print(f"\n[bold]Value: {r['value']}[/bold] ({r['experiment_count']} experiments)")
        for mk, mv in r.get("metrics", {}).items():
            console.print(f"  {mk}: mean={mv['mean']} (n={mv['count']})")


def _load_or_build(store: FileStore, save_path: str | None) -> ExperimentGraph:
    if save_path:
        return ExperimentGraph.load(save_path)
    return ExperimentGraph.build(store)
