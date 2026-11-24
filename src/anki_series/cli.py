"""Punto de entrada de la línea de comandos.

Comandos previstos: extract (F1-F2), build (F3), translate (F4), images (F5), all.
"""

from collections import Counter
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import TypeAdapter

from anki_series import __version__
from anki_series.config import Settings
from anki_series.models import Card, Sentence
from anki_series.subtitles import load_sentences
from anki_series.vocab import KnownDeckError, build_vocabulary, load_known_lemmas

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
        int, typer.Option("--limit", min=0, help="Filas de muestra (oraciones o palabras).")
    ] = 10,
    top: Annotated[int, typer.Option("--top", min=1, help="Cuántos lemas conservar.")] = 1000,
    known_deck: Annotated[
        Path | None,
        typer.Option("--known-deck", help="Mazo .apkg con palabras que ya conoces (se excluyen)."),
    ] = None,
    known_field: Annotated[
        str | None,
        typer.Option(
            "--known-field",
            help="Campo del mazo conocido: índice o nombre. Default: KNOWN_DECK_FIELD o 0.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Ruta del JSON. Default: <data>/output/words.json."),
    ] = None,
) -> None:
    """Extrae oraciones (F1) y el vocabulario top N con ejemplo (F2) → words.json."""
    try:
        sentences = load_sentences(input_path)
    except (FileNotFoundError, ValueError) as exc:
        _fail(exc)

    if dry_run:
        _show_sentences(series, sentences, limit)
        return

    settings = Settings()
    known: set[str] = set()
    if known_deck is not None:
        field = known_field if known_field is not None else settings.known_deck_field
        try:
            known = load_known_lemmas(known_deck, field)
        except (FileNotFoundError, KnownDeckError) as exc:
            _fail(exc)
        typer.echo(f"Mazo conocido: {known_deck.name} · {len(known)} formas/lemas a excluir")

    cards = build_vocabulary(sentences, series, top=top, known_lemmas=known)
    path = output or settings.data_dir / "output" / "words.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_CARDS.dump_json(cards, indent=2))

    typer.echo(f"Serie: {series} · Oraciones: {len(sentences)} · Lemas: {len(cards)}")
    typer.echo(f"Escrito: {path}")
    for card in cards[:limit]:
        typer.echo(
            f"  {card.rank:>4}. {card.lemma}/{card.pos} ×{card.frequency}  {card.example_en}"
        )


_CARDS = TypeAdapter(list[Card])


def _fail(exc: Exception) -> NoReturn:
    typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from exc


def _show_sentences(series: str, sentences: list[Sentence], limit: int) -> None:
    per_file = Counter(s.source.file for s in sentences)
    typer.echo(f"Serie: {series}")
    typer.echo(f"Archivos: {len(per_file)} · Oraciones: {len(sentences)}")
    for file, count in per_file.items():
        typer.echo(f"  {file}: {count}")
    if limit and sentences:
        typer.echo(f"\nPrimeras {min(limit, len(sentences))} oraciones:")
        for s in sentences[:limit]:
            typer.echo(f"  [{s.source.file} {s.source.start}] {s.text}")
