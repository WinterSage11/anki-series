"""Fase 2: vocabulario. Oraciones → top N lemas (lema + POS) con una oración de ejemplo.

Flujo: oraciones (F1) → spaCy (lema, POS, NER) → filtros → conteo por (lema, POS) →
exclusión de lemas ya conocidos (mazo .apkg local) → ranking → ``Card`` (solo inglés).
"""

import html
import json
import re
import sqlite3
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import spacy
import zstandard
from spacy.language import Language

from anki_series.models import Card, Sentence

SPACY_MODEL = "en_core_web_sm"

# Categorías que nunca son tarjeta: nombres propios, números, puntuación, símbolos, basura.
EXCLUDED_POS = {"PROPN", "NUM", "PUNCT", "SYM", "X", "SPACE"}
# La stoplist de spaCy (pensada para búsqueda) incluye verbos y sustantivos clave del
# diálogo: get, go, make, take, say, see, front, side... Solo se aplica fuera de estas
# categorías, así se van pronombres, auxiliares, preposiciones y "really/very/just".
CONTENT_POS = {"NOUN", "VERB", "ADJ"}
# Lema válido: solo letras, con apóstrofo o guion interno ("y'all", "t-shirt").
_VALID_LEMMA = re.compile(r"^[a-z]+(?:['-][a-z]+)*$")

# Ejemplo ideal: ni muy corto (poco contexto) ni muy largo (difícil de leer en la tarjeta).
_IDEAL_WORDS = 10
_MIN_WORDS, _MAX_WORDS = 4, 20


@lru_cache(maxsize=1)
def load_nlp() -> Language:
    """Carga spaCy una sola vez. El parser no hace falta: las oraciones ya vienen de F1."""
    return spacy.load(SPACY_MODEL, disable=["parser"])


# --- Extracción y conteo ---------------------------------------------------------------


@dataclass
class _Candidate:
    frequency: int
    example: Sentence
    example_score: tuple[int, int]


def _example_score(sentence: Sentence) -> tuple[int, int]:
    """Menor es mejor: primero que esté en rango, luego cercanía al largo ideal."""
    n = len(sentence.text.split())
    out_of_range = 0 if _MIN_WORDS <= n <= _MAX_WORDS else 1
    return (out_of_range, abs(n - _IDEAL_WORDS))


def _token_key(token, stop_words: set[str]) -> tuple[str, str] | None:
    """(lema, POS) si el token cuenta como vocabulario; ``None`` si se filtra."""
    if token.pos_ in EXCLUDED_POS or token.ent_type_ == "PERSON":
        return None
    lemma = token.lemma_.lower()
    is_stop = token.is_stop or lemma in stop_words
    if is_stop and token.pos_ not in CONTENT_POS:
        return None
    if not _VALID_LEMMA.match(lemma) or len(lemma) < 2:
        return None
    return lemma, token.pos_


def count_lemmas(sentences: list[Sentence], nlp: Language | None = None) -> dict:
    """Cuenta (lema, POS) en todas las oraciones y guarda el mejor ejemplo de cada uno.

    Devuelve ``{(lema, pos): _Candidate}``. Un lema cuenta una vez por cada aparición.
    En empate de calidad de ejemplo gana la primera aparición (orden de la serie).
    """
    nlp = nlp or load_nlp()
    stop_words = nlp.Defaults.stop_words
    candidates: dict[tuple[str, str], _Candidate] = {}
    texts = (s.text for s in sentences)
    for sentence, doc in zip(sentences, nlp.pipe(texts, batch_size=256), strict=True):
        score = _example_score(sentence)
        for token in doc:
            key = _token_key(token, stop_words)
            if key is None:
                continue
            cand = candidates.get(key)
            if cand is None:
                candidates[key] = _Candidate(1, sentence, score)
            else:
                cand.frequency += 1
                if score < cand.example_score:
                    cand.example, cand.example_score = sentence, score
    return candidates


def build_vocabulary(
    sentences: list[Sentence],
    series: str,
    *,
    top: int = 1000,
    known_lemmas: set[str] | None = None,
    nlp: Language | None = None,
) -> list[Card]:
    """Top ``top`` lemas (lema + POS) que no están en ``known_lemmas``, como ``Card``.

    La exclusión es por lema sin importar el POS: si conoces "store", se quitan
    ``store/NOUN`` y ``store/VERB``. Orden: frecuencia desc., luego lema y POS (estable).
    """
    known = known_lemmas or set()
    candidates = count_lemmas(sentences, nlp)
    ranked = sorted(
        ((key, c) for key, c in candidates.items() if key[0] not in known),
        key=lambda item: (-item[1].frequency, item[0][0], item[0][1]),
    )
    return [
        Card(
            series=series,
            lemma=lemma,
            pos=pos,
            rank=rank,
            frequency=cand.frequency,
            example_en=cand.example.text,
            source=cand.example.source,
        )
        for rank, ((lemma, pos), cand) in enumerate(ranked[:top], start=1)
    ]


# --- Mazo conocido (.apkg) -------------------------------------------------------------


class KnownDeckError(ValueError):
    """El .apkg no se pudo leer o el campo elegido no contiene palabras."""


# Prioridad: el formato más nuevo primero. Cuando existe uno nuevo, collection.anki2 es
# solo un stub de compatibilidad ("actualiza Anki") y no trae las notas reales.
_COLLECTIONS = ("collection.anki21b", "collection.anki21", "collection.anki2")
_TAG = re.compile(r"<[^>]+>")
_SOUND = re.compile(r"\[sound:[^\]]*\]")


def strip_html(value: str) -> str:
    value = _SOUND.sub(" ", value)
    value = _TAG.sub(" ", value)
    value = html.unescape(value).replace("\xa0", " ")
    return " ".join(value.split())


def _extract_collection(apkg: Path, dest: Path) -> Path:
    """Copia la colección SQLite del .apkg a ``dest`` (descomprimiendo zstd si aplica)."""
    try:
        zf = zipfile.ZipFile(apkg)
    except zipfile.BadZipFile as exc:
        raise KnownDeckError(f"No es un .apkg válido (zip): {apkg}") from exc
    with zf:
        names = set(zf.namelist())
        name = next((n for n in _COLLECTIONS if n in names), None)
        if name is None:
            raise KnownDeckError(f"El .apkg no contiene una colección de Anki: {apkg}")
        out = dest / "collection.sqlite"
        with zf.open(name) as src, out.open("wb") as dst:
            if name.endswith(".anki21b"):
                zstandard.ZstdDecompressor().copy_stream(src, dst)
            else:
                dst.write(src.read())
    return out


def _field_names(db: sqlite3.Connection) -> dict[int, list[str]]:
    """Nombres de campos por tipo de nota. Esquema nuevo (tabla ``fields``) o viejo (JSON)."""
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    names: dict[int, list[str]] = {}
    if "fields" in tables:
        for ntid, _ord, name in db.execute("SELECT ntid, ord, name FROM fields ORDER BY ntid, ord"):
            names.setdefault(ntid, []).append(name)
    else:
        models = json.loads(db.execute("SELECT models FROM col").fetchone()[0] or "{}")
        for mid, model in models.items():
            names[int(mid)] = [f["name"] for f in model["flds"]]
    return names


def _resolve_index(field: int | str, names: list[str]) -> int | None:
    if isinstance(field, int):
        return field
    if field.isdigit():
        return int(field)
    lowered = [n.lower() for n in names]
    return lowered.index(field.lower()) if field.lower() in lowered else None


def read_known_words(apkg: Path, field: int | str = 0) -> list[str]:
    """Lee el texto (sin HTML) del campo ``field`` de cada nota del .apkg.

    ``field`` puede ser un índice (0, 1, ...) o el nombre del campo ("Palabra"). Si el
    campo no parece contener palabras (p. ej. un índice numérico), lanza ``KnownDeckError``
    con la lista de campos disponibles para elegir el correcto.
    """
    if not apkg.is_file():
        raise FileNotFoundError(apkg)
    with tempfile.TemporaryDirectory() as tmp:
        db_path = _extract_collection(apkg, Path(tmp))
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            names = _field_names(db)
            words: list[str] = []
            for mid, flds in db.execute("SELECT mid, flds FROM notes"):
                idx = _resolve_index(field, names.get(mid, []))
                values = flds.split("\x1f")
                if idx is not None and idx < len(values):
                    text = strip_html(values[idx])
                    if text:
                        words.append(text)
        finally:
            db.close()

    alphabetic = sum(any(ch.isalpha() for ch in w) for w in words)
    if not words or alphabetic < len(words) / 2:
        available = "; ".join(
            ", ".join(f"{i}={n!r}" for i, n in enumerate(fields)) for fields in names.values()
        )
        raise KnownDeckError(
            f"El campo {field!r} de {apkg.name} no parece contener palabras. "
            f"Campos disponibles: {available}"
        )
    return words


def known_lemmas_from_words(words: Iterable[str], nlp: Language | None = None) -> set[str]:
    """Lematiza las palabras del mazo conocido.

    Solo usa entradas de una palabra: saber "give up" no implica saber "give". Guarda la
    forma tal cual y su lema ("left" → {"left", "leave"}) para no depender del contexto.
    """
    nlp = nlp or load_nlp()
    singles = sorted({w.lower() for w in words if len(w.split()) == 1})
    lemmas: set[str] = set(singles)
    for doc in nlp.pipe(singles, batch_size=512):
        lemmas.update(t.lemma_.lower() for t in doc if not t.is_punct)
    return lemmas


def load_known_lemmas(apkg: Path, field: int | str = 0, nlp: Language | None = None) -> set[str]:
    return known_lemmas_from_words(read_known_words(apkg, field), nlp)
