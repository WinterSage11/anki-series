"""Punto de entrada de la línea de comandos.

Comandos previstos: extract (F1-F2), build (F3), translate (F4), images (F5), all.
"""

import typer

from anki_series import __version__

app = typer.Typer(help="Genera mazos de Anki a partir de subtítulos de una serie.")


@app.callback()
def main() -> None:
    """anki-series: subtítulos .srt → mazo .apkg."""


@app.command()
def version() -> None:
    """Muestra la versión instalada."""
    typer.echo(__version__)
