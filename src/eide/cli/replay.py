from __future__ import annotations

import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eide.replay.engine import ReplayEngine
from eide.storage.filestore import FileStore

console = Console()
DEFAULT_EIDE_ROOT = os.environ.get("EIDE_ROOT", ".eide")


@click.command()
@click.argument("experiment_id")
@click.option("--no-env", is_flag=True, help="Skip environment restoration")
@click.option("--no-data", is_flag=True, help="Skip data restoration")
@click.option("--no-save", is_flag=True, help="Don't save replay result")
@click.option("--artifacts", default=None, help="Artifacts root path")
@click.option("--work-dir", default=None, help="Working directory for replay")
@click.option("--root", default=None, help="EIDE store root")
def replay(
    experiment_id: str,
    no_env: bool,
    no_data: bool,
    no_save: bool,
    artifacts: str | None,
    work_dir: str | None,
    root: str | None,
):
    """Replay an experiment and verify reproducibility"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)

    try:
        experiment = store.load(experiment_id)
    except FileNotFoundError:
        console.print(f"[red]✖ Experiment {experiment_id} not found[/red]")
        return

    console.print(Panel(f"[bold cyan]Replay Experiment[/bold cyan]\n{experiment_id}"))

    engine = ReplayEngine(
        store=store,
        artifacts_root=artifacts,
        work_dir=work_dir,
    )

    try:
        result = engine.replay(
            experiment,
            restore_environment=not no_env,
            restore_data=not no_data,
            save_replay=not no_save,
        )
    finally:
        if work_dir is None:
            engine.cleanup()

    verdict = result.verdict
    if verdict.is_identical:
        console.print(f"\n[green]{verdict.summary}[/green]")
    else:
        console.print(f"\n[red]{verdict.summary}[/red]")

    console.print("\n[bold]Replay Result:[/bold]")
    console.print(f"  Replay ID: {result.replay_experiment.id}")
    console.print(f"  Total changes: {verdict.total_changes}")
    console.print(f"  Output changes: {verdict.output_changes}")
    console.print(f"  Environment changes: {verdict.env_changes}")
    console.print(f"  Structure changes: {verdict.struct_changes}")

    if result.step_results:
        console.print("\n[bold]Step Results:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Step")
        table.add_column("Duration")
        table.add_column("Status")
        for sr in result.step_results:
            status = "[green]✔[/green]" if sr.get("success") else "[red]✖[/red]"
            table.add_row(
                sr["step_name"],
                f"{sr.get('duration_seconds', 0):.1f}s",
                status,
            )
        console.print(table)

    if verdict.diverged:
        console.print("\n[bold yellow]Divergence Details:[/bold yellow]")
        for c in verdict.report.changes[:10]:
            console.print(f"  • {c.description}")
        if len(verdict.report.changes) > 10:
            console.print(f"  ... and {len(verdict.report.changes) - 10} more")
