from __future__ import annotations

import os

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--root", default=os.environ.get("EIDE_ROOT", ".eide"), help="EIDE store root")
@click.option("--host", default="127.0.0.1", help="Host to bind")
@click.option("--port", default=8080, type=int, help="Port to bind")
def serve(root: str, host: str, port: int):
    """Start the MCP server"""
    try:
        from eide.mcp.server import serve as _serve

        console.print(f"[green]Starting EIDE MCP server on {host}:{port}...[/green]")
        console.print(f"[dim]Store root: {root}[/dim]")
        _serve(eide_root=root, host=host, port=port)
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")
        raise
