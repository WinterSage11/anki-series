"""Fase 5: imágenes de Pixabay (o locales) para las tarjetas ilustrables.

Flujo por tarjeta: override manual (``overrides.yaml``) > ``card.image.query`` del LLM →
caché por query → búsqueda en Pixabay → filtro de relevancia → descarga → JPEG de máx.
400 px en ``data/cache/images/``. Mejor sin imagen que con una engañosa: si ningún
resultado pasa el filtro, la tarjeta queda ``NOT_FOUND`` y el proceso sigue.

El índice del caché se guarda (atómico) después de cada búsqueda: al reanudar no se repite
ninguna llamada a la API.
"""

import hashlib
import io
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from anki_series.models import Card, ImageStatus

PIXABAY_URL = "https://pixabay.com/api/"
MAX_SIZE = 400  # px del lado más largo; la tarjeta mide 400 px de ancho
JPEG_QUALITY = 80
PER_PAGE = 5  # resultados a revisar por búsqueda (Pixabay exige 3–200)


class ImageError(RuntimeError):
    """Error no recuperable de Pixabay (API key inválida, reintentos agotados…)."""


class OverridesError(ValueError):
    """``overrides.yaml`` con formato inválido."""


# --- Pixabay ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    """Resultado de búsqueda de Pixabay (solo lo que usamos)."""

    image_url: str  # webformatURL: ~640 px, suficiente para redimensionar a 400
    page_url: str
    user: str
    tags: str


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def is_relevant(query: str, tags: str) -> bool:
    """¿Comparte el resultado alguna palabra con la query?

    Pixabay ordena por popularidad y a veces devuelve fotos que no tienen que ver. Se pide
    que al menos una palabra de la query (≥ 3 letras) aparezca en los tags, comparando por
    prefijo para cubrir plurales y compuestos ("car" ~ "cars", "police" ~ "policeman").
    """
    query_words = {w for w in _words(query) if len(w) >= 3} or _words(query)
    tag_words = _words(tags)
    return any(t.startswith(q) or q.startswith(t) for q in query_words for t in tag_words)


class PixabayClient:
    """Cliente mínimo de Pixabay con reintentos ante errores transitorios.

    Reintenta con backoff exponencial (2 s, 4 s, 8 s…) ante red/timeout, 429 y 5xx. En un
    429 respeta ``X-RateLimit-Reset`` si viene (límite: 100 peticiones por minuto).
    """

    def __init__(
        self,
        http: httpx.Client,
        api_key: str,
        *,
        max_retries: int = 3,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.http = http
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff = backoff
        self.sleep = sleep
        self.requests = 0  # llamadas a la API de búsqueda (para el resumen)

    def _get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            wait = self.backoff * 2**attempt
            try:
                response = self.http.get(url, params=params)
            except httpx.TransportError as exc:  # conexión, timeout…
                # Sin str(exc): podría incluir la URL con la API key.
                problem = type(exc).__name__
            else:
                if response.status_code == 429 or response.status_code >= 500:
                    problem = f"HTTP {response.status_code}"
                    reset = response.headers.get("X-RateLimit-Reset")
                    if response.status_code == 429 and reset and reset.isdigit():
                        wait = max(wait, float(reset))
                else:
                    return response
            if attempt == self.max_retries:
                attempts = self.max_retries + 1
                raise ImageError(f"Error transitorio tras {attempts} intentos: {problem}")
            self.sleep(wait)
        raise AssertionError("unreachable")

    def search(self, query: str) -> list[Hit]:
        """Resultados relevantes para ``query`` (puede ser lista vacía)."""
        params = {
            "key": self.api_key,
            "q": query,
            "lang": "en",
            "image_type": "photo",
            "safesearch": "true",
            "per_page": PER_PAGE,
        }
        self.requests += 1
        response = self._get(PIXABAY_URL, params)
        if response.status_code != 200:
            # Pixabay responde texto plano, p. ej. "[ERROR 400] Invalid or missing API key".
            raise ImageError(f"Pixabay respondió {response.status_code}: {response.text[:200]}")
        hits = []
        for raw in response.json().get("hits", []):
            hit = Hit(
                image_url=raw.get("webformatURL", ""),
                page_url=raw.get("pageURL", ""),
                user=raw.get("user", ""),
                tags=raw.get("tags", ""),
            )
            if hit.image_url and is_relevant(query, hit.tags):
                hits.append(hit)
        return hits

    def download(self, url: str) -> bytes | None:
        """Bytes de la imagen; ``None`` si ya no existe (4xx): se prueba el siguiente hit."""
        response = self._get(url)
        return response.content if response.status_code == 200 else None


# --- Redimensionado --------------------------------------------------------------------


def save_jpeg(source: bytes | Path, dest: Path, max_size: int = MAX_SIZE) -> Path:
    """Guarda ``source`` como JPEG optimizado con el lado más largo ≤ ``max_size``."""
    raw = source.read_bytes() if isinstance(source, Path) else source
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img = ImageOps.exif_transpose(img)  # respeta la orientación de la cámara
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                background = Image.new("RGB", img.size, "white")  # JPEG no tiene alfa
                background.paste(img, mask=img.getchannel("A"))
                img = background
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError(f"No es una imagen válida: {exc}") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(dest)
    return dest


# --- Caché -----------------------------------------------------------------------------


class ImageCache:
    """Imágenes ya procesadas por query, en ``<dir>/<sha1>.jpg`` + ``<dir>/index.json``.

    El índice guarda también las búsquedas sin resultado (``found: false``) para no repetirlas.
    Para reintentar una query, borra su entrada del índice.
    """

    def __init__(self, directory: Path) -> None:
        self.dir = directory
        self.index_path = directory / "index.json"
        self._data: dict[str, dict[str, Any]] = {}
        if self.index_path.is_file():
            self._data = json.loads(self.index_path.read_text(encoding="utf-8"))

    @staticmethod
    def normalize(query: str) -> str:
        return " ".join(query.lower().split())

    def path_for(self, query: str) -> Path:
        digest = hashlib.sha1(self.normalize(query).encode()).hexdigest()
        return self.dir / f"{digest}.jpg"

    def get(self, query: str) -> dict[str, Any] | None:
        entry = self._data.get(self.normalize(query))
        if entry and entry.get("found") and not Path(entry["path"]).is_file():
            return None  # el archivo se borró: hay que volver a descargar
        return entry

    def put(self, query: str, entry: dict[str, Any]) -> None:
        self._data[self.normalize(query)] = entry

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.index_path)


# --- Overrides -------------------------------------------------------------------------


class Override(BaseModel):
    """Ajuste manual de una palabra: exactamente uno de ``query``, ``image`` o ``skip``."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = None  # otra búsqueda en Pixabay
    image: Path | None = None  # imagen local (ruta relativa al directorio del proyecto)
    skip: bool = False  # sin imagen

    @model_validator(mode="after")
    def _exactly_one(self) -> "Override":
        if sum([bool(self.query and self.query.strip()), self.image is not None, self.skip]) != 1:
            raise ValueError("usa exactamente una de: query, image, skip: true")
        return self


def _override_key(key: str) -> str:
    lemma, _, pos = str(key).strip().partition("/")
    return f"{lemma.strip().lower()}/{pos.strip().upper()}" if pos else lemma.strip().lower()


def load_overrides(path: Path) -> dict[str, Override]:
    """Lee ``overrides.yaml``; llaves ``lema`` (todos sus POS) o ``lema/POS``. Sin archivo → {}."""
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise OverridesError(f"{path}: YAML inválido: {exc}") from exc
    if not isinstance(raw, dict):
        raise OverridesError(f"{path}: se esperaba un mapeo `palabra: {{...}}`")
    overrides = {}
    for key, value in raw.items():
        try:
            overrides[_override_key(key)] = Override.model_validate(value)
        except ValidationError as exc:
            errors = "; ".join(e["msg"] for e in exc.errors())
            raise OverridesError(f"{path}: entrada `{key}`: {errors}") from exc
    return overrides


def find_override(overrides: dict[str, Override], card: Card) -> Override | None:
    """``lema/POS`` gana sobre ``lema``."""
    return overrides.get(f"{card.lemma}/{card.pos}") or overrides.get(card.lemma)


# --- Orquestación ----------------------------------------------------------------------


@dataclass
class ImageStats:
    found: int = 0
    not_found: int = 0
    skipped: int = 0
    from_cache: int = 0
    keys: list[str] = field(default_factory=list)  # tarjetas procesadas en esta corrida


def needs_work(card: Card, overrides: dict[str, Override]) -> bool:
    """Overrides siempre; si no, solo tarjetas ``pending`` ya traducidas."""
    if find_override(overrides, card) is not None:
        return True
    return card.image.status == ImageStatus.PENDING and card.imageable is not None


def _with_image(card: Card, **update: Any) -> Card:
    blank = {"path": None, "source": None, "source_url": None, "credit": None, "reason": None}
    return card.model_copy(update={"image": card.image.model_copy(update=blank | update)})


def _from_query(
    card: Card,
    query: str,
    cache: ImageCache,
    client: Callable[[], PixabayClient],
    stats: ImageStats,
) -> Card:
    entry = cache.get(query)
    if entry is not None:
        stats.from_cache += 1
    else:
        entry = {"found": False}
        pixabay = client()
        for hit in pixabay.search(query):
            data = pixabay.download(hit.image_url)
            if data is None:
                continue
            path = save_jpeg(data, cache.path_for(query))
            entry = {
                "found": True,
                "path": str(path),
                "source_url": hit.page_url,
                "credit": f"{hit.user} / Pixabay" if hit.user else "Pixabay",
            }
            break
        cache.put(query, entry)
        cache.save()

    if not entry["found"]:
        return _with_image(
            card, status=ImageStatus.NOT_FOUND, query=query, reason="sin resultados relevantes"
        )
    return _with_image(
        card,
        status=ImageStatus.FOUND,
        query=query,
        path=entry["path"],
        source="pixabay",
        source_url=entry["source_url"],
        credit=entry["credit"],
    )


def process_card(
    card: Card,
    *,
    cache: ImageCache,
    overrides: dict[str, Override],
    client: Callable[[], PixabayClient],
    stats: ImageStats,
) -> Card:
    """Decide y aplica la imagen de una tarjeta (ver ``needs_work``)."""
    override = find_override(overrides, card)
    if override is not None and override.skip:
        return _with_image(card, status=ImageStatus.SKIPPED, reason="override: skip")
    if override is not None and override.image is not None:
        if not override.image.is_file():
            raise OverridesError(f"Imagen de override no encontrada: {override.image}")
        digest = hashlib.sha1(card.key.encode()).hexdigest()[:12]
        path = save_jpeg(override.image, cache.dir / f"override-{digest}.jpg")
        return _with_image(
            card, status=ImageStatus.FOUND, path=str(path), source="local", credit="override"
        )
    if override is not None and override.query:
        return _from_query(card, override.query.strip(), cache, client, stats)
    if not card.imageable or not card.image.query:
        return _with_image(card, status=ImageStatus.SKIPPED, reason="no ilustrable")
    return _from_query(card, card.image.query, cache, client, stats)


def fetch_images(
    cards: Sequence[Card],
    *,
    cache: ImageCache,
    overrides: dict[str, Override],
    client_factory: Callable[[], PixabayClient],
    limit: int | None = None,
    checkpoint: Callable[[list[Card]], None] | None = None,
    checkpoint_every: int = 10,
    on_card: Callable[[int, int, Card], None] | None = None,
) -> tuple[list[Card], ImageStats]:
    """Procesa las tarjetas pendientes (o con override); devuelve todas las tarjetas.

    ``client_factory`` solo se llama si hay que buscar en Pixabay: con el caché completo no
    hace falta API key. ``checkpoint(tarjetas)`` se llama cada ``checkpoint_every`` tarjetas y
    también si hay un error o Ctrl+C, para poder reanudar.
    """
    out = list(cards)
    todo = [i for i, c in enumerate(out) if needs_work(c, overrides)]
    if limit is not None:
        todo = todo[:limit]
    stats = ImageStats()
    client: PixabayClient | None = None

    def get_client() -> PixabayClient:
        nonlocal client
        if client is None:
            client = client_factory()
        return client

    try:
        for n, i in enumerate(todo, start=1):
            card = process_card(
                out[i], cache=cache, overrides=overrides, client=get_client, stats=stats
            )
            out[i] = card
            stats.keys.append(card.key)
            match card.image.status:
                case ImageStatus.FOUND:
                    stats.found += 1
                case ImageStatus.NOT_FOUND:
                    stats.not_found += 1
                case _:
                    stats.skipped += 1
            if on_card:
                on_card(n, len(todo), card)
            if checkpoint and n % checkpoint_every == 0:
                checkpoint(out)
    except BaseException:
        if checkpoint:
            checkpoint(out)
        raise
    return out, stats
