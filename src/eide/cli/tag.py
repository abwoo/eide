from __future__ import annotations

import os

import click
from rich.console import Console
from rich.table import Table

from eide.storage.filestore import FileStore

console = Console()
DEFAULT_EIDE_ROOT = os.environ.get("EIDE_ROOT", ".eide")


@click.group()
def tag():
    """Manage experiment tags"""


@tag.command("add")
@click.argument("experiment_id")
@click.argument("key")
@click.argument("value")
@click.option("--root", default=None, help="EIDE store root")
def tag_add(experiment_id: str, key: str, value: str, root: str | None):
    """Add a tag to an experiment"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    try:
        exp = store.load(experiment_id)
    except FileNotFoundError:
        console.print(f"[red]✖ Experiment [bold]{experiment_id}[/bold] not found[/red]")
        return
    exp.tags[key] = value
    store.save(exp)
    console.print(
        f"[green]✔ Tag [bold]{key}={value}[/bold] added to [bold]{experiment_id}[/bold][/green]"
    )


@tag.command("remove")
@click.argument("experiment_id")
@click.argument("key")
@click.option("--root", default=None, help="EIDE store root")
def tag_remove(experiment_id: str, key: str, root: str | None):
    """Remove a tag from an experiment"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    try:
        exp = store.load(experiment_id)
    except FileNotFoundError:
        console.print(f"[red]✖ Experiment [bold]{experiment_id}[/bold] not found[/red]")
        return
    if key in exp.tags:
        old = exp.tags.pop(key)
        store.save(exp)
        console.print(
            (
                f"[green]✔ Tag [bold]{key}={old}[/bold]"
                f" removed from [bold]{experiment_id}[/bold][/green]"
            )
        )
    else:
        console.print(
            f"[yellow]✖ Tag [bold]{key}[/bold] not found on [bold]{experiment_id}[/bold][/yellow]"
        )


@tag.command("list")
@click.argument("experiment_id")
@click.option("--root", default=None, help="EIDE store root")
def tag_list(experiment_id: str, root: str | None):
    """List all tags for an experiment"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    try:
        exp = store.load(experiment_id)
    except FileNotFoundError:
        console.print(f"[red]✖ Experiment [bold]{experiment_id}[/bold] not found[/red]")
        return
    if not exp.tags:
        console.print(f"[yellow]No tags on experiment [bold]{experiment_id}[/bold][/yellow]")
        return
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Key")
    table.add_column("Value")
    for k, v in sorted(exp.tags.items()):
        table.add_row(k, v)
    console.print(table)
