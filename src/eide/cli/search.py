from __future__ import annotations

import itertools
import os

import click

from eide.storage.filestore import FileStore


@click.command()
@click.argument("query")
@click.option("--field", default="all", help="Search field: name, tag, param, metadata, id, all")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--root", default=None, help="EIDE store root")
def search(query: str, field: str, as_json: bool, root: str | None):
    """Search experiments by keyword across all fields"""
    store = FileStore(root or os.environ.get("EIDE_ROOT", ".eide"))
    results = store.search(query, field=field)

    if as_json:
        import json as j

        click.echo(j.dumps([r.model_dump(mode="json") for r in results], indent=2, default=str))
        return

    if not results:
        click.echo(f"No experiments found matching '{query}'")
        return

    click.echo(f"Found {len(results)} experiment(s) matching '{query}':")
    click.echo("")
    for exp in results:
        name = exp.name or "(unnamed)"
        tags = ", ".join(f"{k}={v}" for k, v in itertools.islice(exp.tags.items(), 3))
        metrics = ", ".join(
            f"{k}={v}"
            for k, v in (
                list(exp.outputs.metrics.items()) if exp.outputs and exp.outputs.metrics else []
            )[:3]
        )
        click.echo(f"  {exp.id[:12]}  {name}")
        if tags:
            click.echo(f"       tags: {tags}")
        if metrics:
            click.echo(f"       metrics: {metrics}")
