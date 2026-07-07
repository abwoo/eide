from __future__ import annotations

import os

import click
from rich.console import Console
from rich.panel import Panel

from eide.causal.attribution import CausalAttribution
from eide.storage.filestore import FileStore

console = Console()
DEFAULT_EIDE_ROOT = os.environ.get("EIDE_ROOT", ".eide")


@click.command()
@click.argument("experiment_a")
@click.argument("experiment_b")
@click.option("--root", default=None, help="EIDE store root")
def explain(experiment_a: str, experiment_b: str, root: str | None):
    """Explain why two experiments differ"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    try:
        exp_a = store.load(experiment_a)
        exp_b = store.load(experiment_b)
    except FileNotFoundError as e:
        console.print(f"[red]✖ {e}[/red]")
        return

    causal = CausalAttribution(store=store)
    explanation = causal.explain(exp_a, exp_b)

    console.print(
        Panel(f"[bold cyan]Experiment Explanation[/bold cyan]\n{experiment_a} vs {experiment_b}")
    )
    console.print(f"\n[bold]Summary:[/bold]\n{explanation['summary']}")

    stats = explanation["change_statistics"]
    console.print("\n[bold]Change Statistics:[/bold]")
    for k, v in stats.items():
        console.print(f"  {k}: {v}")

    console.print("\n[bold]Likely Causes (ranked):[/bold]")
    for i, cause in enumerate(explanation["likely_cause"], 1):
        confidence_pct = cause["confidence"] * 100
        console.print(
            f"  {i}. [{'green' if cause['confidence'] > 0.6 else 'yellow'}]{cause['cause']}[/]"
            f" ({confidence_pct:.0f}% confidence)"
        )
        console.print(f"     {cause['reasoning']}")

    changelog = explanation.get("changelog", [])
    if isinstance(changelog, str):
        changelog = changelog.split("\n")
    if changelog:
        console.print("\n[bold]Changes Log:[/bold]")
        for line in changelog:
            console.print(f"  • {line}")
