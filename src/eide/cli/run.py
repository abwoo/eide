from __future__ import annotations

import os
from pathlib import Path

import click
import yaml
from rich.console import Console

from eide.capture.manual import capture_manual
from eide.storage.filestore import FileStore

console = Console()


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--root", default=None, help="EIDE store root")
@click.option("--diff", is_flag=True, help="Auto-diff against most similar baseline")
@click.option("--dry-run", is_flag=True, help="Parse and validate config without saving")
def run(config_file: str, root: str | None, diff: bool, dry_run: bool):
    """Run an experiment defined in a YAML config file"""
    raw = yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))

    exp = capture_manual(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        project=raw.get("project", ""),
        parameters=raw.get("parameters", {}),
        metrics={k: float(v) for k, v in raw.get("metrics", {}).items()},
        tags=raw.get("tags", {}),
        pipeline_steps=raw.get("pipeline", {}).get("steps")
        if isinstance(raw.get("pipeline"), dict)
        else raw.get("pipeline"),
        data_versions=raw.get("data_versions", {}),
        decisions=raw.get("decisions", {}),
        metadata=raw.get("metadata", {}),
    )

    if dry_run:
        console.print(f"[dim]Dry-run: would create experiment [bold]{exp.id}[/bold][/dim]")
        console.print(f"  Name: {exp.name or '(unnamed)'}")
        console.print(f"  Project: {exp.project or '(none)'}")
        console.print(f"  Parameters: {len(exp.parameters)}")
        console.print(f"  Pipeline steps: {len(exp.pipeline.steps)}")
        return

    store = FileStore(root or os.environ.get("EIDE_ROOT", ".eide"))
    store.save(exp)
    console.print(
        f"[green]✔ Created experiment [bold]{exp.id}[/bold] from [bold]{config_file}[/bold][/green]"
    )

    if diff:
        result = store.diff_with_baseline(exp.id)
        if result:
            console.print(
                f"\n[bold]Auto-diff with baseline [dim]{result['baseline_id'][:12]}[/dim]:[/bold]"
            )
            console.print(f"  [dim]{result['summary']}[/dim]")
