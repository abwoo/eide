from __future__ import annotations

import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eide.diff.engine import DiffEngine
from eide.storage.filestore import FileStore

console = Console()
DEFAULT_EIDE_ROOT = os.environ.get("EIDE_ROOT", ".eide")


@click.command()
@click.argument("experiment_a")
@click.argument("experiment_b")
@click.option("--root", default=None, help="EIDE store root")
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
@click.option("--artifacts", default=None, help="Artifacts root path for figure comparison")
def diff(experiment_a: str, experiment_b: str, root: str | None, output_format: str, artifacts: str | None):
    """Diff two experiments"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)

    try:
        exp_a = store.load(experiment_a)
        exp_b = store.load(experiment_b)
    except FileNotFoundError as e:
        console.print(f"[red]✖ {e}[/red]")
        return

    engine = DiffEngine(artifacts_root=artifacts)
    report = engine.diff(exp_a, exp_b)

    if output_format == "json":
        console.print(report.model_dump_json(indent=2))
        return

    _print_text_report(report, experiment_a, experiment_b)


def _print_text_report(report, exp_a_id: str, exp_b_id: str):
    console.print(Panel(f"[bold cyan]Experiment Diff Report[/bold cyan]\n{exp_a_id} vs {exp_b_id}"))

    if not report.changes:
        console.print("[green]No differences found.[/green]")
        return

    console.print(f"\n[bold]Summary:[/bold] {len(report.changes)} difference(s)")

    for cat in ["parameter", "structural", "environment", "output", "semantic", "data_version"]:
        cat_changes = [c for c in report.changes if c.category.value == cat]
        if not cat_changes:
            continue

        console.print(f"\n[bold yellow]{cat.upper()}[/bold yellow]")
        table = Table(show_header=True, header_style="dim")
        table.add_column("Type", width=10)
        table.add_column("Key", width=40)
        table.add_column("Description")
        for c in cat_changes:
            type_str = {"added": "[green]+add[/green]", "removed": "[red]-del[/red]", "modified": "[blue]~mod[/blue]"}.get(
                c.change_type.value, c.change_type.value
            )
            table.add_row(type_str, c.key, c.description)
        console.print(table)

    if report.impact_estimates:
        console.print("\n[bold]Impact Estimates:[/bold]")
        for ie in report.impact_estimates:
            console.print(f"  {ie.metric}: {ie.change_percent:.1f}% {ie.change_direction}")
