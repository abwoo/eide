from __future__ import annotations

import click


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind")
@click.option("--port", default=8765, type=int, help="Port to bind")
@click.option("--root", default=None, help="EIDE store root")
def dashboard(host: str, port: int, root: str | None):
    """Start the EIDE web dashboard"""
    try:
        from eide.dashboard.app import serve
        serve(host=host, port=port, root=root)
    except ImportError as e:
        import click
        click.echo(f"Missing dependencies: {e}")
        click.echo("Install: pip install eide-core[full]")
    except Exception as e:
        import click
        click.echo(f"Failed to start dashboard: {e}")
        raise
