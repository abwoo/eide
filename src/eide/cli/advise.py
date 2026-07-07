from __future__ import annotations

import os

import click
from rich.console import Console
from rich.panel import Panel

from eide.intelligence.advisor import ExperimentAdvisor
from eide.storage.filestore import FileStore

console = Console()


def _get_root(root: str | None) -> str:
    return root or os.environ.get("EIDE_ROOT", ".eide")


@click.group()
def advise():
    """Intelligent experiment analysis"""


@advise.command()
@click.argument("experiment_a")
@click.argument("experiment_b")
@click.option("--root", default=None, help="EIDE store root")
def compare(experiment_a: str, experiment_b: str, root: str | None):
    """Compare two experiments with intelligent insights"""
    store = FileStore(_get_root(root))
    try:
        exp_a = store.load(experiment_a)
        exp_b = store.load(experiment_b)
    except FileNotFoundError as e:
        console.print(f"[red]✖ {e}[/red]")
        return

    advisor = ExperimentAdvisor(store=store)
    insight = advisor.compare(exp_a, exp_b)

    console.print(
        Panel(
            f"[bold cyan]Experiment Comparison Insight[/bold cyan]"
            f"\n{experiment_a} vs {experiment_b}"
        )
    )
    console.print(f"\n[bold]Summary:[/bold] {insight.diff_summary}")

    if insight.likely_causes:
        console.print("\n[bold]Likely Causes:[/bold]")
        for cause in insight.likely_causes:
            pct = cause["confidence"] * 100
            console.print(
                f"  • [{'green' if pct > 60 else 'yellow'}]{cause['cause']}[/] ({pct:.0f}%)"
            )
            console.print(f"    {cause['reasoning']}")

    if insight.recommended_actions:
        console.print("\n[bold]Recommended Actions:[/bold]")
        for action in insight.recommended_actions:
            p_style = {"high": "red", "medium": "yellow", "low": "dim"}.get(
                action["priority"], "dim"
            )
            console.print(
                f"  [{p_style}]{action['type']}[/] ({action['priority']}): {action['message']}"
            )

    if insight.impact_estimates:
        console.print("\n[bold]Impact Estimates:[/bold]")
        for ie in insight.impact_estimates:
            console.print(f"  {ie.metric}: {ie.change_percent:.1f}% {ie.change_direction}")


@advise.command()
@click.argument("experiment_id")
@click.option("--root", default=None, help="EIDE store root")
def risk(experiment_id: str, root: str | None):
    """Assess risk of an experiment"""
    store = FileStore(_get_root(root))
    try:
        exp = store.load(experiment_id)
    except FileNotFoundError:
        console.print(f"[red]✖ Experiment {experiment_id} not found[/red]")
        return

    advisor = ExperimentAdvisor(store=store)
    assessment = advisor.assess_risk(exp)

    level_colors = {"low": "green", "medium": "yellow", "high": "red", "unknown": "dim"}
    console.print(
        f"\n[bold]Risk Level:[/bold] [{level_colors.get(assessment.risk_level, 'dim')}]"
        f"{assessment.risk_level.upper()}[/]"
    )

    if assessment.risks:
        console.print("\n[bold]Risk Details:[/bold]")
        for risk in assessment.risks:
            sev_color = {"info": "dim", "warning": "yellow", "critical": "red"}.get(
                risk["severity"], "dim"
            )
            console.print(f"  [{sev_color}]• {risk['message']}[/]")

    recommendation = advisor.recommend_baseline(exp)
    if recommendation.best_match:
        console.print(
            f"\n[bold]Recommended Baseline:[/bold] {recommendation.best_match['experiment_id']}"
        )
        console.print(f"  Similarity: {recommendation.best_match['similarity']:.1%}")
    elif recommendation.message:
        console.print(f"\n[dim]{recommendation.message}[/dim]")


@advise.command()
@click.argument("experiment_id")
@click.option("--root", default=None, help="EIDE store root")
def baseline(experiment_id: str, root: str | None):
    """Find and recommend the best baseline for comparison"""
    store = FileStore(_get_root(root))
    try:
        exp = store.load(experiment_id)
    except FileNotFoundError:
        console.print(f"[red]✖ Experiment {experiment_id} not found[/red]")
        return

    advisor = ExperimentAdvisor(store=store)
    recommendation = advisor.recommend_baseline(exp)

    if recommendation.best_match:
        console.print(f"[bold cyan]Baseline Recommendation for {experiment_id}[/bold cyan]")
        console.print(f"\n  Best match: [bold]{recommendation.best_match['experiment_id']}[/bold]")
        console.print(f"  Name: {recommendation.best_match.get('name', '(unnamed)')}")
        console.print(f"  Similarity: {recommendation.best_match['similarity']:.1%}")

        if recommendation.significant_differences:
            console.print("\n[bold]Key differences:[/bold]")
            for diff in recommendation.significant_differences[:5]:
                console.print(f"  • {diff.description}")
    elif recommendation.message:
        console.print(f"[yellow]{recommendation.message}[/yellow]")
