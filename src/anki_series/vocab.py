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
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import spacy
import zstandard
from spacy.language import Language
from wordfreq import iter_wordlist

from anki_series.models import Card, Sentence

SPACY_MODEL = "en_core_web_sm"

# Categorías que nunca son tarjeta: nombres propios, números, puntuación, símbolos, basura
# e interjecciones (yeah, oh, okay, uh, hey: relleno sin valor como tarjeta).
EXCLUDED_POS = {"PROPN", "NUM", "PUNCT", "SYM", "X", "SPACE", "INTJ"}
# Muletillas vocales que spaCy no siempre etiqueta como INTJ ("Mm-hmm" → mm/NOUN +
# hmm/NOUN; "Mm, good" → mm/ADJ). Se aceptan alargadas ("hmmm", "uhh"), salvo "er":
# "err" es un verbo real.
_FILLERS = {"hmm", "mm", "mhm", "uh", "um", "huh", "ah", "er"}
_FILLER_ELONGATED = re.compile(r"^(?:h*m+|m+h+m+|u+h+|u+m+|h+u+h+|a+h+)$")
# Señales de "nombre" para la votación por lema (ver ``count_lemmas``).
_NAME_POS = "PROPN"
_NAME_ENTS = {"PERSON"}
# Contracciones informales → (lema, POS) del verbo que representan. spaCy parte "gonna"
# en "gon" + "na" (y a veces etiqueta "gon" como NOUN); "wanna" queda con lema "wanna".
_CONTRACTIONS = {
    "gonna": ("go", "VERB"),
    "gotta": ("get", "VERB"),
    "wanna": ("want", "VERB"),
}
_CONTRACTION_SPLITS = {("gon", "na"): "gonna", ("got", "ta"): "gotta", ("wan", "na"): "wanna"}
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


@lru_cache(maxsize=1)
def load_nlp_parser() -> Language:
    """spaCy con parser de dependencias (``prt``): solo para detectar phrasal verbs."""
    return spacy.load(SPACY_MODEL)


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


def contraction(token) -> tuple[str, int] | None:
    """Si el token es (o empieza) una contracción informal: (forma, n.º de tokens).

    Reconoce el token único ("wanna") y el par que deja spaCy ("gon" + "na", sin espacio).
    """
    text = token.text.lower()
    if text in _CONTRACTIONS:
        return text, 1
    if token.whitespace_ == "" and token.i + 1 < len(token.doc):
        form = _CONTRACTION_SPLITS.get((text, token.doc[token.i + 1].text.lower()))
        if form:
            return form, 2
    return None


def token_lemma_pos(token) -> tuple[str, str]:
    """(lema en minúsculas, POS) corrigiendo contracciones: "gonna" → ("go", "VERB")."""
    found = contraction(token)
    if found:
        return _CONTRACTIONS[found[0]]
    return token.lemma_.lower(), token.pos_


def is_filler(word: str) -> bool:
    """¿Es una muletilla vocal (hmm, mm, uh, um…)? ``word`` en minúsculas."""
    return word in _FILLERS or bool(_FILLER_ELONGATED.match(word))


def _token_key(token, stop_words: set[str]) -> tuple[str, str] | None:
    """(lema, POS) si el token cuenta como vocabulario; ``None`` si se filtra."""
    if token.pos_ in EXCLUDED_POS and contraction(token) is None:
        return None
    if token.ent_type_ in _NAME_ENTS:
        return None
    lemma, pos = token_lemma_pos(token)
    if is_filler(lemma) or is_filler(token.text.lower()):
        return None
    is_stop = token.is_stop or lemma in stop_words
    if is_stop and pos not in CONTENT_POS:
        return None
    if not _VALID_LEMMA.match(lemma) or len(lemma) < 2:
        return None
    return lemma, pos


def _is_name(token) -> bool:
    return token.pos_ == _NAME_POS or token.ent_type_ in _NAME_ENTS


def count_lemmas(sentences: list[Sentence], nlp: Language | None = None) -> dict:
    """Cuenta (lema, POS) en todas las oraciones y guarda el mejor ejemplo de cada uno.

    Devuelve ``{(lema, pos): _Candidate}``. Un lema cuenta una vez por cada aparición.
    En empate de calidad de ejemplo gana la primera aparición (orden de la serie).

    Nombres propios colados: spaCy etiqueta un mismo personaje como ``PROPN`` en unas
    líneas y como ``NOUN`` en otras ("Harvey"). Por eso cada lema vota en todo el corpus:
    si más de la mitad de sus apariciones son ``PROPN`` o entidad ``PERSON``, se descarta
    en todos sus POS.
    """
    nlp = nlp or load_nlp()
    stop_words = nlp.Defaults.stop_words
    candidates: dict[tuple[str, str], _Candidate] = {}
    name_votes: Counter[str] = Counter()
    total_votes: Counter[str] = Counter()
    texts = (s.text for s in sentences)
    for sentence, doc in zip(sentences, nlp.pipe(texts, batch_size=256), strict=True):
        score = _example_score(sentence)
        for token in doc:
            if token.is_alpha and contraction(token) is None:
                surface = token.lemma_.lower()
                total_votes[surface] += 1
                if _is_name(token):
                    name_votes[surface] += 1
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
    names = {lemma for lemma, n in name_votes.items() if n * 2 > total_votes[lemma]}
    return {key: c for key, c in candidates.items() if key[0] not in names}


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


# --- Palabras ultra-comunes (wordfreq) -------------------------------------------------


def common_words(n: int) -> set[str]:
    """Las ``n`` palabras más frecuentes del inglés según ``wordfreq`` (datos locales).

    Solo cuenta palabras (se saltan números y contracciones como "don't"), así ``n``
    significa n palabras. Se comparan contra el lema tal cual, sin lematizar la lista:
    las formas base de las palabras comunes ya están en ella, y lematizar palabras sueltas
    sin contexto mete ruido ("saw" → "see").
    """
    words: set[str] = set()
    if n <= 0:
        return words
    for word in iter_wordlist("en"):
        if word.isascii() and word.isalpha():
            words.add(word)
            if len(words) >= n:
                break
    return words


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


# Marcadores que los mazos ponen en expresiones ("figure sth out", "to give up").
_EXPR_PLACEHOLDERS = {"sth", "smth", "something", "sb", "someone", "somebody", "one's"}
_PARENS = re.compile(r"\([^)]*\)|\[[^\]]*\]")


def known_expressions_from_words(words: Iterable[str], nlp: Language | None = None) -> set[str]:
    """Expresiones (2+ palabras) del mazo conocido, tal cual y lematizadas.

    "Gave up" → {"gave up", "give up"}; se quitan "to " inicial, paréntesis y marcadores
    como "sth"/"someone" ("figure (sth) out" → "figure out").
    """
    nlp = nlp or load_nlp()
    entries: set[str] = set()
    for word in words:
        text = _PARENS.sub(" ", word.lower())
        tokens = [t for t in re.findall(r"[a-z]+(?:'[a-z]+)?", text) if t not in _EXPR_PLACEHOLDERS]
        if tokens[:1] == ["to"]:
            tokens = tokens[1:]
        if len(tokens) >= 2:
            entries.add(" ".join(tokens))
    found = set(entries)
    for doc in nlp.pipe(sorted(entries), batch_size=512):
        found.add(" ".join(t.lemma_.lower() for t in doc))
    return found


def load_known_lemmas(apkg: Path, field: int | str = 0, nlp: Language | None = None) -> set[str]:
    return known_lemmas_from_words(read_known_words(apkg, field), nlp)
