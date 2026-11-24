"""Fase 4: traducción contextual al español de México/LATAM con OpenAI.

Flujo: tarjetas (solo inglés) → lotes pequeños → Structured Outputs (Pydantic) → caché en disco
→ tarjetas con ``translation``, ``example_es``, ``imageable`` e ``image.query``.

El caché se guarda después de cada lote: si el proceso se interrumpe o falla a la mitad,
al volver a correrlo solo se pagan los lotes que faltan.
"""

import hashlib
import json
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import openai
from pydantic import BaseModel, Field, ValidationError

from anki_series.models import Card

# Cambiar este número invalida el caché (úsalo si cambias el prompt o el esquema).
PROMPT_VERSION = 4

SYSTEM_PROMPT = """\
You help a native Spanish speaker from Mexico learn English vocabulary from TV subtitles.
For each item you get an English word (lemma), its part of speech (spaCy universal tag)
and a real sentence from the show where it appears. Return, for every item, using its same id:

- phrase: FIRST decide if, in this sentence, the word is part of a phrasal verb or idiom.
  If so, write that expression in base form (e.g. "give up", "look after"); otherwise null.
- phrase_translation: if phrase is not null, the Mexican/LATAM Spanish meaning of the whole
  expression (give up → rendirse); otherwise null.
- translation: the Spanish (Mexico/Latin America) translation of the WORD ALONE. The card
  teaches the single word, not the expression: if phrase is not null, translate the word's
  own most common meaning (give → dar, look → mirar), NEVER the meaning of the phrase.
  If phrase is null, translate the word as used in the sentence. Keep the part of speech:
  verbs in infinitive, nouns in singular, adjectives in masculine singular. Use 1-3 words;
  you may add a second common meaning after a comma only if it is very frequent. Slang and
  swear words are kept: give the natural Mexican/LATAM equivalent with the same register,
  never a censored or neutral one.
- example_es: a natural, idiomatic translation of the whole sentence into Mexican/LATAM
  Spanish (not word-for-word), as a Mexican would actually say it (e.g. "I'm running low on
  gas" → "ya casi no tengo gasolina"). It does not need to reuse the word translation.
  Keep the tone (informal, rude, sarcastic...).
- imageable: true only if the word itself, with the meaning given in translation, is a
  concrete object, animal, place, person role or physical action that a stock photo can show
  unambiguously (e.g. car, knife, kitchen, police officer, run). Always false for swear words,
  intensifiers, abstract words, function-like words, feelings, time expressions and slang.
  When in doubt, false.
- image_query: if imageable, a short English search query (1-3 words) for a stock photo site
  that depicts ONLY the word itself, disambiguated for this sense (e.g. "bat" the animal →
  "bat animal flying"); never mix in other objects from the sentence. Otherwise null.
"""

RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    openai.APIConnectionError,  # incluye APITimeoutError
    openai.RateLimitError,
    openai.InternalServerError,
)


class TranslationError(RuntimeError):
    """Un lote no se pudo traducir (error no recuperable o reintentos agotados)."""


# --- Esquema de Structured Outputs (lo que el modelo debe devolver) ---


class WordTranslation(BaseModel):
    id: int = Field(description="Same id as the input item.")
    # Va antes de translation a propósito: los campos se generan en orden y así el modelo
    # detecta el phrasal verb ANTES de traducir la palabra suelta. No se copia a la Card.
    phrase: str | None = None
    phrase_translation: str | None = None
    translation: str
    example_es: str
    imageable: bool
    image_query: str | None


class TranslationBatch(BaseModel):
    items: list[WordTranslation]


# --- Caché ---


class TranslationCache:
    """Diccionario ``clave → WordTranslation`` persistido en un JSON.

    La clave depende del modelo, la versión del prompt, el lema, el POS y la oración: si
    cambia cualquiera de ellos se vuelve a traducir. No depende de la serie, así que dos
    series con la misma oración comparten traducción.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        if path.is_file():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(card: Card, model: str) -> str:
        raw = f"v{PROMPT_VERSION}|{model}|{card.lemma}|{card.pos}|{card.example_en}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(self, card: Card, model: str) -> WordTranslation | None:
        entry = self._data.get(self.key(card, model))
        return WordTranslation.model_validate(entry) if entry else None

    def put(self, card: Card, model: str, result: WordTranslation) -> None:
        self._data[self.key(card, model)] = result.model_dump()

    def save(self) -> None:
        """Escritura atómica: un Ctrl+C a la mitad no deja el JSON corrupto."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._data)


# --- Llamada al modelo ---


def request_batch(
    client: Any,
    model: str,
    cards: Sequence[Card],
    *,
    max_retries: int = 3,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[int, WordTranslation]:
    """Traduce un lote; devuelve ``{índice en el lote: resultado}``.

    Reintenta con backoff exponencial (2 s, 4 s, 8 s…) ante errores transitorios
    (red, timeout, 429, 5xx). Otros errores (API key inválida, 400…) fallan de inmediato.
    Los ids que el modelo omita o invente se ignoran; esas tarjetas quedan sin caché
    y se reintentan en la siguiente corrida.
    """
    items = [
        {"id": i, "word": c.lemma, "pos": c.pos, "sentence": c.example_en}
        for i, c in enumerate(cards)
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
    ]

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=TranslationBatch,
                temperature=0.2,
            )
            break
        except RETRYABLE_ERRORS as exc:
            if attempt == max_retries:
                raise TranslationError(
                    f"Error transitorio tras {max_retries + 1} intentos: {exc}"
                ) from exc
            sleep(backoff * 2**attempt)
        except (openai.OpenAIError, ValidationError) as exc:
            raise TranslationError(f"Error de OpenAI: {exc}") from exc

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise TranslationError(f"El modelo se negó a traducir el lote: {message.refusal}")
    parsed: TranslationBatch | None = message.parsed
    if parsed is None:
        raise TranslationError("Respuesta vacía o sin el formato esperado.")

    return {item.id: _normalize(item) for item in parsed.items if 0 <= item.id < len(cards)}


def _normalize(item: WordTranslation) -> WordTranslation:
    """Limpia espacios y garantiza coherencia entre ``imageable`` e ``image_query``."""
    query = (item.image_query or "").strip() or None
    imageable = item.imageable and query is not None
    return item.model_copy(
        update={
            "translation": item.translation.strip(),
            "example_es": item.example_es.strip(),
            "imageable": imageable,
            "image_query": query if imageable else None,
        }
    )


# --- Orquestación ---


def apply_translation(card: Card, result: WordTranslation) -> Card:
    """Llena los campos de F4 sin tocar el resto (la imagen la decide la F5)."""
    image = card.image.model_copy(update={"query": result.image_query})
    return card.model_copy(
        update={
            "translation": result.translation,
            "example_es": result.example_es,
            "imageable": result.imageable,
            "image": image,
        }
    )


def _chunks(seq: Sequence[Card], size: int) -> Iterator[Sequence[Card]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def translate_cards(
    cards: Sequence[Card],
    *,
    cache: TranslationCache,
    model: str,
    client_factory: Callable[[], Any],
    batch_size: int = 20,
    on_batch: Callable[[int, int], None] | None = None,
    **request_kwargs: Any,
) -> tuple[list[Card], int]:
    """Traduce las tarjetas usando el caché primero; devuelve ``(tarjetas, nuevas)``.

    ``client_factory`` solo se llama si hay algo que traducir: con el caché completo
    no hace falta API key. ``on_batch(hechos, total)`` sirve para mostrar progreso.
    Si un lote falla se lanza ``TranslationError``; lo ya traducido quedó en el caché.
    """
    pending = [c for c in cards if cache.get(c, model) is None]
    new = 0
    if pending:
        client = client_factory()
        done = 0
        for batch in _chunks(pending, batch_size):
            results = request_batch(client, model, batch, **request_kwargs)
            for i, result in results.items():
                cache.put(batch[i], model, result)
            cache.save()
            new += len(results)
            done += len(batch)
            if on_batch:
                on_batch(done, len(pending))

    out: list[Card] = []
    for card in cards:
        result = cache.get(card, model)
        # Sin resultado = el modelo omitió la tarjeta; queda sin traducir y se reintenta luego.
        out.append(apply_translation(card, result) if result else card)
    return out, new
