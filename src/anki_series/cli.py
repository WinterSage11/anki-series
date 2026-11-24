"""Punto de entrada de la línea de comandos.

Comandos previstos: extract (F1-F2), build (F3), translate (F4), images (F5), all.
"""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from anki_series import __version__
from anki_series.subtitles import load_sentences

app = typer.Typer(help="Genera mazos de Anki a partir de subtítulos de una serie.")


@app.callback()
def main() -> None:
    """anki-series: subtítulos .srt → mazo .apkg."""


@app.command()
def version() -> None:
    """Muestra la versión instalada."""
    typer.echo(__version__)


@app.command()
def extract(
    input_path: Annotated[
        Path, typer.Argument(help="Archivo .srt o carpeta con .srt (temporada/serie).")
    ],
    series: Annotated[str, typer.Option("--series", help="Nombre de la serie.")],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Solo muestra las oraciones extraídas; no escribe nada."),
    ] = False,
    limit: Annotated[
        int, typer.Option("--limit", min=0, help="Oraciones de muestra en --dry-run.")
    ] = 10,
) -> None:
    """Extrae oraciones de los subtítulos (F1) y el vocabulario (F2, pendiente)."""
    try:
        sentences = load_sentences(input_path)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if not dry_run:
        typer.secho(
            "La extracción de vocabulario (words.json) llega en la fase 2. "
            "Por ahora usa --dry-run.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)

    per_file = Counter(s.source.file for s in sentences)
    typer.echo(f"Serie: {series}")
    typer.echo(f"Archivos: {len(per_file)} · Oraciones: {len(sentences)}")
    for file, count in per_file.items():
        typer.echo(f"  {file}: {count}")
    if limit and sentences:
        typer.echo(f"\nPrimeras {min(limit, len(sentences))} oraciones:")
        for s in sentences[:limit]:
            typer.echo(f"  [{s.source.file} {s.source.start}] {s.text}")
