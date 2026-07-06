from __future__ import annotations

import json
import os
from pathlib import Path

import click
from rich.console import Console

from eide.core.ir import ExperimentIR
from eide.storage.filestore import FileStore

console = Console()
DEFAULT_EIDE_ROOT = os.environ.get("EIDE_ROOT", ".eide")


@click.command()
@click.argument("experiment_id")
@click.argument("output_path", type=click.Path())
@click.option("--root", default=None, help="EIDE store root")
@click.option("--pretty", is_flag=True, default=True, help="Pretty-print JSON")
def export(experiment_id: str, output_path: str, root: str | None, pretty: bool):
    """Export an experiment as a portable JSON file"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    try:
        exp = store.load(experiment_id)
    except FileNotFoundError:
        console.print(f"[red]✖ Experiment [bold]{experiment_id}[/bold] not found[/red]")
        return

    data = exp.model_dump(mode="json")
    indent = 2 if pretty else None
    Path(output_path).write_text(json.dumps(data, indent=indent, default=str), encoding="utf-8")
    console.print(f"[green]✔ Exported [bold]{experiment_id}[/bold] to [bold]{output_path}[/bold][/green]")


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--root", default=None, help="EIDE store root")
@click.option("--id", "override_id", default=None, help="Override experiment ID on import")
def import_exp(input_path: str, root: str | None, override_id: str | None):
    """Import an experiment from a JSON file"""
    store = FileStore(root or DEFAULT_EIDE_ROOT)
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if override_id:
        data["id"] = override_id
    exp = ExperimentIR.model_validate(data)
    path = store.save(exp)
    console.print(f"[green]✔ Imported experiment [bold]{exp.id}[/bold] from [bold]{input_path}[/bold][/green]")
    console.print(f"  Path: {path}")
