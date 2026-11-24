"""Punto de entrada de la línea de comandos.

Flujo: extract (F1-F2) → translate (F4) → images (F5) → audio (F6) → build (F3).
"""

import re
from collections import Counter
from pathlib import Path
from typing import Annotated, NoReturn

import httpx
import typer
from openai import OpenAI
from pydantic import TypeAdapter

from anki_series import __version__
from anki_series.audio import DEFAULT_CONCURRENCY, AudioCache, AudioError, generate_audio
from anki_series.config import Settings
from anki_series.deck import build_deck
from anki_series.images import (
    ImageCache,
    ImageError,
    OverridesError,
    PixabayClient,
    fetch_images,
    load_overrides,
)
from anki_series.models import Card, ImageStatus, Sentence
from anki_series.subtitles import load_sentences
from anki_series.translate import TranslationCache, TranslationError, translate_cards
from anki_series.vocab import (
    KnownDeckError,
    build_vocabulary,
    common_words,
    load_known_lemmas,
)

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
    exclude_common: Annotated[
        int,
        typer.Option(
            "--exclude-common",
            min=0,
            help="Excluye las N palabras más frecuentes del inglés (wordfreq). 0 = no excluir.",
        ),
    ] = 0,
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
    if exclude_common:
        known |= common_words(exclude_common)
        typer.echo(f"Palabras comunes excluidas (wordfreq): top {exclude_common}")

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


@app.command()
def build(
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            "-i",
            help="JSON de tarjetas. Default: <data>/output/cards.json o, si no existe, words.json.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Ruta del .apkg. Default: <data>/output/<serie>.apkg."),
    ] = None,
) -> None:
    """Construye el mazo .apkg (F3) a partir de words.json / cards.json."""
    settings = Settings()
    out_dir = settings.data_dir / "output"
    if input_path is None:
        cards_json = out_dir / "cards.json"
        input_path = cards_json if cards_json.exists() else out_dir / "words.json"
    if not input_path.is_file():
        _fail(FileNotFoundError(f"No existe {input_path}. Corre primero `extract`."))

    cards = _CARDS.validate_json(input_path.read_bytes())
    if not cards:
        _fail(ValueError(f"{input_path} no tiene tarjetas."))
    path = output or out_dir / f"{_slug(cards[0].series)}.apkg"
    try:
        build_deck(cards, path)
    except (FileNotFoundError, ValueError) as exc:
        _fail(exc)

    with_image = sum(1 for c in cards if c.image.status == ImageStatus.FOUND)
    translated = sum(1 for c in cards if c.translation)
    with_audio = sum(1 for c in cards if c.audio.word)
    typer.echo(f"Entrada: {input_path}")
    typer.echo(
        f"Notas: {len(cards)} · Con traducción: {translated} · Con imagen: {with_image}"
        f" · Con audio: {with_audio}"
    )
    typer.echo(f"Escrito: {path}")


@app.command()
def translate(
    input_path: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="JSON de entrada. Default: <data>/output/words.json."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="JSON de salida. Default: <data>/output/cards.json."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", min=1, help="Solo las primeras N tarjetas (para probar). Escribe solo esas."
        ),
    ] = None,
    batch_size: Annotated[
        int, typer.Option("--batch-size", min=1, max=100, help="Tarjetas por llamada.")
    ] = 20,
    model: Annotated[
        str | None, typer.Option("--model", help="Modelo de OpenAI. Default: OPENAI_MODEL.")
    ] = None,
) -> None:
    """Traduce al español (México/LATAM) con OpenAI (F4): words.json → cards.json."""
    settings = Settings()
    out_dir = settings.data_dir / "output"
    input_path = input_path or out_dir / "words.json"
    if not input_path.is_file():
        _fail(FileNotFoundError(f"No existe {input_path}. Corre primero `extract`."))

    cards = _CARDS.validate_json(input_path.read_bytes())
    if limit is not None:
        cards = cards[:limit]
    model = model or settings.openai_model
    cache = TranslationCache(settings.data_dir / "cache" / "translations.json")

    def client_factory() -> OpenAI:
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        if not key:
            _fail(ValueError("Falta OPENAI_API_KEY (ponla en .env)."))
        # max_retries=0: los reintentos los maneja translate.request_batch.
        return OpenAI(api_key=key, max_retries=0)

    def progress(done: int, total: int) -> None:
        typer.echo(f"  lote listo: {done}/{total}")

    typer.echo(f"Entrada: {input_path} · Tarjetas: {len(cards)} · Modelo: {model}")
    try:
        translated, new = translate_cards(
            cards,
            cache=cache,
            model=model,
            client_factory=client_factory,
            batch_size=batch_size,
            on_batch=progress,
        )
    except TranslationError as exc:
        typer.secho(
            "El progreso ya traducido quedó en caché; vuelve a correr el comando para continuar.",
            err=True,
        )
        _fail(exc)

    path = output or out_dir / "cards.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_CARDS.dump_json(translated, indent=2))

    done = sum(1 for c in translated if c.translation)
    imageable = sum(1 for c in translated if c.imageable)
    typer.echo(
        f"Traducidas: {done}/{len(translated)} (nuevas: {new}, de caché: {done - new})"
        f" · Ilustrables: {imageable}"
    )
    if done < len(translated):
        typer.echo(f"  {len(translated) - done} sin traducir; vuelve a correr para reintentarlas.")
    typer.echo(f"Escrito: {path}")


@app.command()
def images(
    input_path: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="JSON de tarjetas. Default: <data>/output/cards.json."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="JSON de salida. Default: el mismo de entrada (se actualiza)."
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", min=1, help="Procesa solo N tarjetas pendientes; escribe todas igual."
        ),
    ] = None,
    overrides_path: Annotated[
        Path | None,
        typer.Option("--overrides", help="Ajustes manuales. Default: <data>/overrides.yaml."),
    ] = None,
) -> None:
    """Busca y descarga imágenes de Pixabay (F5): cards.json → cards.json con imagen."""
    settings = Settings()
    input_path = input_path or settings.data_dir / "output" / "cards.json"
    output = output or input_path
    if not input_path.is_file():
        _fail(FileNotFoundError(f"No existe {input_path}. Corre primero `translate`."))

    cards = _CARDS.validate_json(input_path.read_bytes())
    try:
        overrides = load_overrides(overrides_path or settings.data_dir / "overrides.yaml")
    except OverridesError as exc:
        _fail(exc)
    cache = ImageCache(settings.data_dir / "cache" / "images")
    http: list[httpx.Client] = []

    def client_factory() -> PixabayClient:
        key = settings.pixabay_api_key.get_secret_value() if settings.pixabay_api_key else ""
        if not key:
            _fail(ValueError("Falta PIXABAY_API_KEY (ponla en .env)."))
        http.append(_pixabay_http())
        return PixabayClient(http[0], key)

    def checkpoint(current: list[Card]) -> None:
        _write_json_atomic(output, current)

    def progress(done: int, total: int, card: Card) -> None:
        if done % 25 == 0 or done == total:
            typer.echo(f"  {done}/{total}")

    pending = sum(1 for c in cards if c.image.status == ImageStatus.PENDING)
    typer.echo(
        f"Entrada: {input_path} · Tarjetas: {len(cards)} · Pendientes: {pending}"
        f" · Overrides: {len(overrides)}"
    )
    try:
        updated, stats = fetch_images(
            cards,
            cache=cache,
            overrides=overrides,
            client_factory=client_factory,
            limit=limit,
            checkpoint=checkpoint,
            on_card=progress,
        )
    except (ImageError, OverridesError) as exc:
        typer.secho(
            f"El progreso quedó guardado en {output}; vuelve a correr para continuar.", err=True
        )
        _fail(exc)
    finally:
        for client in http:
            client.close()

    _write_json_atomic(output, updated)
    with_image = sum(1 for c in updated if c.image.status == ImageStatus.FOUND)
    left = sum(1 for c in updated if c.image.status == ImageStatus.PENDING and c.imageable)
    typer.echo(
        f"Procesadas: {len(stats.keys)} · Con imagen: {stats.found} · Sin resultado:"
        f" {stats.not_found} · Omitidas: {stats.skipped} · De caché: {stats.from_cache}"
    )
    typer.echo(f"Total con imagen: {with_image}/{len(updated)}")
    if left:
        typer.echo(f"  {left} ilustrables pendientes; vuelve a correr para continuar.")
    typer.echo(f"Escrito: {output}")


@app.command()
def audio(
    input_path: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="JSON de tarjetas. Default: <data>/output/cards.json."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="JSON de salida. Default: el mismo de entrada (se actualiza)."
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Procesa solo N tarjetas sin audio; escribe todas."),
    ] = None,
    voice: Annotated[
        str | None, typer.Option("--voice", help="Voz de edge-tts. Default: TTS_VOICE.")
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", min=1, max=16, help="Audios generados en paralelo.")
    ] = DEFAULT_CONCURRENCY,
) -> None:
    """Genera audio .mp3 con voces neuronales (F6): palabra + oración de ejemplo."""
    settings = Settings()
    input_path = input_path or settings.data_dir / "output" / "cards.json"
    output = output or input_path
    if not input_path.is_file():
        _fail(FileNotFoundError(f"No existe {input_path}. Corre primero `translate`."))

    cards = _CARDS.validate_json(input_path.read_bytes())
    voice = voice or settings.tts_voice
    cache = AudioCache(settings.data_dir / "cache" / "audio")

    def checkpoint(current: list[Card]) -> None:
        _write_json_atomic(output, current)

    def progress(done: int, total: int, card: Card) -> None:
        if done % 25 == 0 or done == total:
            typer.echo(f"  {done}/{total}")

    typer.echo(f"Entrada: {input_path} · Tarjetas: {len(cards)} · Voz: {voice}")
    try:
        updated, stats = generate_audio(
            cards,
            cache=cache,
            voice=voice,
            limit=limit,
            concurrency=concurrency,
            checkpoint=checkpoint,
            on_card=progress,
        )
    except AudioError as exc:
        typer.secho(
            f"El progreso quedó guardado en {output}; vuelve a correr para continuar.", err=True
        )
        _fail(exc)

    _write_json_atomic(output, updated)
    with_audio = sum(1 for c in updated if c.audio.voice == voice and c.audio.word)
    typer.echo(
        f"Procesadas: {len(stats.keys)} · Audios nuevos: {stats.generated}"
        f" · De caché: {stats.from_cache}"
    )
    typer.echo(f"Total con audio ({voice}): {with_audio}/{len(updated)}")
    if with_audio < len(updated):
        typer.echo(f"  {len(updated) - with_audio} sin audio; vuelve a correr para continuar.")
    typer.echo(f"Escrito: {output}")


@app.command("all")
def run_all(
    input_path: Annotated[
        Path, typer.Argument(help="Archivo .srt o carpeta con .srt (temporada/serie).")
    ],
    series: Annotated[str, typer.Option("--series", help="Nombre de la serie.")],
    top: Annotated[int, typer.Option("--top", min=1, help="Cuántos lemas conservar.")] = 1000,
    known_deck: Annotated[
        Path | None,
        typer.Option("--known-deck", help="Mazo .apkg con palabras que ya conoces (se excluyen)."),
    ] = None,
    known_field: Annotated[
        str | None,
        typer.Option("--known-field", help="Campo del mazo conocido: índice o nombre."),
    ] = None,
    exclude_common: Annotated[
        int,
        typer.Option(
            "--exclude-common", min=0, help="Excluye las N palabras más frecuentes del inglés."
        ),
    ] = 0,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit", min=1, help="Solo las primeras N palabras en todo el flujo (para probar)."
        ),
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Modelo de OpenAI. Default: OPENAI_MODEL.")
    ] = None,
    voice: Annotated[
        str | None, typer.Option("--voice", help="Voz de edge-tts. Default: TTS_VOICE.")
    ] = None,
    overrides_path: Annotated[
        Path | None,
        typer.Option("--overrides", help="Ajustes de imágenes. Default: <data>/overrides.yaml."),
    ] = None,
    concurrency: Annotated[
        int, typer.Option("--concurrency", min=1, max=16, help="Audios generados en paralelo.")
    ] = DEFAULT_CONCURRENCY,
) -> None:
    """Flujo completo: extract → translate → images → audio → build.

    Usa los cachés de cada etapa: tras un error, vuelve a correrlo y solo hará lo que falta.
    """
    steps = ("extract", "translate", "images", "audio", "build")

    def step(n: int) -> None:
        typer.secho(f"\n[{n}/{len(steps)}] {steps[n - 1]}", bold=True)

    step(1)
    extract(
        input_path,
        series=series,
        dry_run=False,
        limit=0,
        top=top,
        known_deck=known_deck,
        known_field=known_field,
        exclude_common=exclude_common,
        output=None,
    )
    step(2)
    # --limit recorta aquí: cards.json queda con N tarjetas y las etapas siguientes
    # procesan todo cards.json.
    translate(input_path=None, output=None, limit=limit, batch_size=20, model=model)
    step(3)
    images(input_path=None, output=None, limit=None, overrides_path=overrides_path)
    step(4)
    audio(input_path=None, output=None, limit=None, voice=voice, concurrency=concurrency)
    step(5)
    build(input_path=None, output=None)


_CARDS = TypeAdapter(list[Card])


def _pixabay_http() -> httpx.Client:
    """Cliente HTTP de Pixabay (los tests lo reemplazan por uno con MockTransport)."""
    return httpx.Client(timeout=30, follow_redirects=True)


def _write_json_atomic(path: Path, cards: list[Card]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(_CARDS.dump_json(cards, indent=2))
    tmp.replace(path)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "deck"


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
