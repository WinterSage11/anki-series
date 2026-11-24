"""Fase 6: audio con voces neuronales (edge-tts) para la palabra y la oración de ejemplo.

Por tarjeta se generan dos ``.mp3``: el lema y ``example_en`` en texto plano. Cada audio se
guarda en ``data/cache/audio/<sha1(voz, texto)>.mp3``: el mismo texto con la misma voz nunca
se vuelve a pedir (p. ej. ``run/VERB`` y ``run/NOUN`` comparten el audio de "run"), y cambiar
de voz o de oración genera un archivo nuevo sin tocar los anteriores.

Las llamadas corren en paralelo con un límite (``asyncio.Semaphore``), se reintentan con
backoff ante errores transitorios y el ``cards.json`` se guarda (atómico) cada N tarjetas y
ante error/Ctrl+C, para poder reanudar.
"""

import asyncio
import hashlib
import html
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp  # dependencia de edge-tts; solo para reconocer sus errores de red
import edge_tts
from edge_tts.exceptions import EdgeTTSException

from anki_series.models import AudioInfo, Card

DEFAULT_CONCURRENCY = 4  # conexiones simultáneas al servicio de voz

# Errores que vale la pena reintentar: el servicio no respondió bien o se cortó la red.
# Un ValueError (p. ej. voz con formato inválido) no se reintenta.
TRANSIENT_ERRORS = (EdgeTTSException, aiohttp.ClientError, TimeoutError, ConnectionError)

# (texto, voz, ruta_destino) → escribe el .mp3. Los tests inyectan uno falso.
Synthesizer = Callable[[str, str, Path], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]


class AudioError(RuntimeError):
    """Error no recuperable al generar audio (voz inválida, reintentos agotados…)."""


def plain_text(text: str) -> str:
    """Texto para leer en voz alta: sin etiquetas HTML, con entidades resueltas y sin
    espacios repetidos."""
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(without_tags).split())


async def edge_tts_synthesize(text: str, voice: str, dest: Path) -> None:
    """Sintetizador real: una conexión a edge-tts por texto."""
    await edge_tts.Communicate(text, voice).save(str(dest))


# --- Caché -----------------------------------------------------------------------------


class AudioCache:
    """``.mp3`` por (voz, texto) en ``<dir>/<sha1>.mp3``. El archivo mismo es el caché:
    para regenerar un audio, bórralo."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def path_for(self, voice: str, text: str) -> Path:
        digest = hashlib.sha1(f"{voice}\x1f{plain_text(text)}".encode()).hexdigest()
        return self.dir / f"{digest}.mp3"

    @staticmethod
    def has(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0


# --- Síntesis con reintentos -----------------------------------------------------------


async def synthesize_with_retry(
    synthesize: Synthesizer,
    text: str,
    voice: str,
    dest: Path,
    *,
    max_retries: int = 3,
    backoff: float = 2.0,
    sleep: Sleep = asyncio.sleep,
) -> Path:
    """Genera ``dest`` (escritura atómica vía ``.tmp``); backoff 2 s, 4 s, 8 s…"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    for attempt in range(max_retries + 1):
        try:
            await synthesize(text, voice, tmp)
            if not AudioCache.has(tmp):
                raise edge_tts.exceptions.NoAudioReceived("el archivo quedó vacío")
        except TRANSIENT_ERRORS as exc:
            problem = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise AudioError(f"No se pudo generar el audio de {text!r} ({voice}): {exc}") from exc
        else:
            tmp.replace(dest)
            return dest
        tmp.unlink(missing_ok=True)
        if attempt == max_retries:
            raise AudioError(
                f"Error transitorio tras {max_retries + 1} intentos con {text!r}: {problem}"
            )
        await sleep(backoff * 2**attempt)
    raise AssertionError("unreachable")


# --- Orquestación ----------------------------------------------------------------------


@dataclass
class AudioStats:
    generated: int = 0  # .mp3 nuevos (llamadas exitosas al servicio)
    from_cache: int = 0  # audios que ya existían en el caché
    keys: list[str] = field(default_factory=list)  # tarjetas procesadas en esta corrida


def expected_audio(card: Card, voice: str, cache: AudioCache) -> AudioInfo:
    """Rutas que deberían tener los audios de ``card`` con ``voice``."""
    example = plain_text(card.example_en)
    return AudioInfo(
        voice=voice,
        word=str(cache.path_for(voice, card.lemma)),
        example=str(cache.path_for(voice, example)) if example else None,
    )


def needs_work(card: Card, voice: str, cache: AudioCache) -> bool:
    """Sin audio, con otra voz/texto o con el .mp3 borrado del disco."""
    expected = expected_audio(card, voice, cache)
    if card.audio != expected:
        return True
    return not all(AudioCache.has(Path(p)) for p in (expected.word, expected.example) if p)


def generate_audio(
    cards: Sequence[Card],
    *,
    cache: AudioCache,
    voice: str,
    synthesize: Synthesizer = edge_tts_synthesize,
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    checkpoint: Callable[[list[Card]], None] | None = None,
    checkpoint_every: int = 10,
    on_card: Callable[[int, int, Card], None] | None = None,
    max_retries: int = 3,
    backoff: float = 2.0,
    sleep: Sleep | None = None,
) -> tuple[list[Card], AudioStats]:
    """Genera los audios que falten; devuelve todas las tarjetas (las demás sin cambios).

    ``limit`` procesa solo las primeras N tarjetas que necesitan audio. ``checkpoint`` se
    llama cada ``checkpoint_every`` tarjetas terminadas y también ante error o Ctrl+C.
    """
    return asyncio.run(
        _generate(
            list(cards),
            cache=cache,
            voice=voice,
            synthesize=synthesize,
            limit=limit,
            concurrency=concurrency,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            on_card=on_card,
            max_retries=max_retries,
            backoff=backoff,
            sleep=sleep or asyncio.sleep,
        )
    )


async def _generate(
    out: list[Card],
    *,
    cache: AudioCache,
    voice: str,
    synthesize: Synthesizer,
    limit: int | None,
    concurrency: int,
    checkpoint: Callable[[list[Card]], None] | None,
    checkpoint_every: int,
    on_card: Callable[[int, int, Card], None] | None,
    max_retries: int,
    backoff: float,
    sleep: Sleep,
) -> tuple[list[Card], AudioStats]:
    todo = [i for i, c in enumerate(out) if needs_work(c, voice, cache)]
    if limit is not None:
        todo = todo[:limit]
    stats = AudioStats()
    semaphore = asyncio.Semaphore(concurrency)
    # Un solo trabajo por archivo: si dos tarjetas comparten texto, la segunda espera al mismo.
    jobs: dict[Path, asyncio.Task[None]] = {}

    async def make(text: str, path: Path) -> None:
        async with semaphore:
            await synthesize_with_retry(
                synthesize, text, voice, path, max_retries=max_retries, backoff=backoff, sleep=sleep
            )
        stats.generated += 1

    def ensure(text: str, path: Path) -> asyncio.Task[None] | None:
        if path in jobs:
            return jobs[path]
        if AudioCache.has(path):
            stats.from_cache += 1
            return None
        jobs[path] = asyncio.create_task(make(text, path))
        return jobs[path]

    async def process(i: int) -> tuple[int, Card]:
        card = out[i]
        audio = expected_audio(card, voice, cache)
        pending = [ensure(card.lemma, Path(audio.word or ""))]
        if audio.example:
            pending.append(ensure(plain_text(card.example_en), Path(audio.example)))
        await asyncio.gather(*(t for t in pending if t is not None))
        return i, card.model_copy(update={"audio": audio})

    tasks = [asyncio.create_task(process(i)) for i in todo]
    try:
        for n, next_done in enumerate(asyncio.as_completed(tasks), start=1):
            i, card = await next_done
            out[i] = card
            stats.keys.append(card.key)
            if on_card:
                on_card(n, len(todo), card)
            if checkpoint and n % checkpoint_every == 0:
                checkpoint(out)
    except BaseException:
        # Error o Ctrl+C: se cancela lo pendiente y se guarda lo que ya terminó.
        pending_tasks = [*tasks, *jobs.values()]
        for task in pending_tasks:
            task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)
        if checkpoint:
            checkpoint(out)
        raise
    return out, stats
