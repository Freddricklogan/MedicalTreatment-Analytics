"""Command-line entry point: `pymaceuticals report --out dist`."""

from __future__ import annotations

from pathlib import Path

import typer

from .report import write_report

app = typer.Typer(add_completion=False, help="Pymaceuticals mouse-study report.")


@app.callback()
def main() -> None:
    """Pymaceuticals mouse-study report."""


@app.command()
def report(
    out: Path = typer.Option(Path("dist"), help="Output directory for the static report."),
    meta: Path = typer.Option(Path("data") / "Mouse_metadata.csv", help="Mouse metadata CSV."),
    results: Path = typer.Option(Path("data") / "Study_results.csv", help="Study results CSV."),
) -> None:
    """Clean, analyse and write the report."""
    path = write_report(out, meta, results)
    print(f"wrote {path}")


if __name__ == "__main__":
    app()
