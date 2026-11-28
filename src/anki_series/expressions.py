"""Expresiones multipalabra: phrasal verbs y modismos ("figure out", "make sense").

Flujo híbrido (solo con ``extract --expressions N``):

1. spaCy con parser → phrasal verbs (``VERB`` + hijo ``prt``, separables incluidos:
   "figure it out" → "figure out") y n-gramas de lemas frecuentes (2 a 4 palabras).
2. Filtro semántico con OpenAI (lotes de ``EXPR_BATCH_SIZE``): descarta combinaciones
   literales o ruido ("sit down", "going to the") y normaliza/clasifica el resto.
   Caché por candidato en disco, guardado tras cada lote.
3. Tarjetas ``Card`` con ``lemma`` = expresión y ``pos`` = ``phrasal`` | ``idiom``.
"""

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from spacy.language import Language
from spacy.tokens import Doc

from anki_series.models import Card, Sentence
from anki_series.translate import parse_with_retries
from anki_series.vocab import (
    _example_score,
    contraction,
    is_filler,
    load_nlp,
    load_nlp_parser,
)

# Cambiar este número invalida el caché del filtro (si cambias su prompt o esquema).
EXPR_FILTER_VERSION = 2
EXPR_BATCH_SIZE = 80
MAX_CANDIDATES = 400
# Presupuesto propio de phrasal verbs dentro de MAX_CANDIDATES: si no, los n-gramas ruidosos
# muy frecuentes desplazan a phrasal verbs de frecuencia 3-6.
MAX_PHRASAL_CANDIDATES = 200
MIN_PHRASAL_FREQ = 2
MIN_NGRAM_FREQ = 3
NGRAM_SIZES = (2, 3, 4)
# Tokens que puede haber entre el verbo y su partícula ("pick the kids up").
MAX_GAP = 3

_NAME_ENTS = {"PERSON"}
# Un n-grama que termina así casi nunca es una expresión ("going to the", "want to be").
_BAD_LAST_POS = {"DET", "CCONJ", "SCONJ", "AUX"}
# "hook up with" es variante de "hook up": se queda solo la más frecuente (ver
# ``_drop_preposition_variants``).
_VARIANT_PREPS = {"with", "to", "of", "on"}

FILTER_PROMPT = """\
You curate vocabulary cards for a native Spanish speaker learning English from TV subtitles.
You get candidate multiword expressions found frequently in a show, each with an id, the
candidate text (lemmatized), a hint ("phrasal" if a parser found verb + particle, "ngram"
otherwise) and one real sentence where it appears. For every item return its same id and:

- keep: true ONLY if the candidate is (or clearly contains) an English phrasal verb or idiom
  with a unitary, non-literal meaning that a learner would not get from its words alone
  (figure out, get rid of, hook up, run over, put up with, make sense, no offense, come on,
  what the hell). false for literal combinations whose meaning is the sum of its words
  (sit down, go home, want to be, look at), sentence fragments or noise (going to the,
  I do not, in the car), names and plain grammar. Also false for incomplete fragments
  that only make sense inside a longer phrase (way out of, the hell out, kind of a) and for
  literal prepositional phrases that are not idioms by themselves (at the altar, in the
  morning, at the bar).
- expression: if keep, the expression in lowercase base form, as a dictionary would list
  it: verbs in infinitive, no subjects or objects (figures it out → "figure out",
  get rid → "get rid of" if the sentence shows that). Only words that appear in the
  sentence, in the same order. Otherwise null.
- kind: if keep, "phrasal" for phrasal verbs (verb + particle/preposition) or "idiom" for
  any other idiom or set phrase. Otherwise null.
"""


class ExpressionError(RuntimeError):
    """El filtro de expresiones falló (error no recuperable o reintentos agotados)."""


# --- Candidatos ------------------------------------------------------------------------


@dataclass
class Candidate:
    text: str  # lemas separados por espacio: "figure out", "make sense"
    hint: Literal["phrasal", "ngram"]
    frequency: int = 0
    example: Sentence | None = None
    example_score: tuple[int, int] = (99, 99)
    names: int = field(default=0, repr=False)  # apariciones con un nombre propio

    def add(self, sentence: Sentence) -> None:
        self.frequency += 1
        score = _example_score(sentence)
        if self.example is None or score < self.example_score:
            self.example, self.example_score = sentence, score


def _lemma(token) -> str:
    return token.lemma_.lower()


def _phrasal_parts(verb) -> list:
    """Verbo + partícula(s) ``prt`` (+ ``prep`` justo después de la partícula)."""
    particles = [c for c in verb.children if c.dep_ == "prt" and c.i > verb.i]
    if not particles:
        return []
    last = particles[-1]
    preps = [c for c in verb.children if c.dep_ == "prep" and c.i == last.i + 1 and c.is_alpha]
    return [verb, *particles, *preps]


def phrasal_candidates(docs: Iterable[tuple[Sentence, Doc]]) -> dict[str, Candidate]:
    """Phrasal verbs según el parser: "figure it out" y "figure out" → "figure out"."""
    found: dict[str, Candidate] = {}
    for sentence, doc in docs:
        for token in doc:
            if token.pos_ != "VERB":
                continue
            parts = _phrasal_parts(token)
            if not parts:
                continue
            text = " ".join(_lemma(t) for t in parts)
            found.setdefault(text, Candidate(text, "phrasal")).add(sentence)
    return {k: c for k, c in found.items() if c.frequency >= MIN_PHRASAL_FREQ}


def _ngram_token_ok(token) -> bool:
    """Un token puede formar parte de un n-grama (la puntuación o un número lo corta)."""
    return (
        token.is_alpha
        and contraction(token) is None
        and not is_filler(token.text.lower())
        and not is_filler(_lemma(token))
    )


def ngram_candidates(
    docs: Iterable[tuple[Sentence, Doc]], stop_words: set[str]
) -> dict[str, Candidate]:
    """2/3/4-gramas de lemas contiguos con ≥ ``MIN_NGRAM_FREQ`` apariciones, sin ruido obvio.

    Se descartan: n-gramas con nombre propio, solo stopwords, terminados en determinante,
    conjunción o auxiliar, y los que solo aparecen dentro de uno más largo con la misma
    frecuencia ("get rid" si siempre es "get rid of").
    """
    found: dict[str, Candidate] = {}
    for sentence, doc in docs:
        seen: set[str] = set()  # cuenta una vez por oración
        for n in NGRAM_SIZES:
            for start in range(len(doc) - n + 1):
                span = doc[start : start + n]
                if not all(_ngram_token_ok(t) for t in span):
                    continue
                if span[-1].pos_ in _BAD_LAST_POS:
                    continue
                lemmas = [_lemma(t) for t in span]
                if all(t.is_stop or lem in stop_words for t, lem in zip(span, lemmas, strict=True)):
                    continue
                text = " ".join(lemmas)
                if text in seen:
                    continue
                seen.add(text)
                cand = found.setdefault(text, Candidate(text, "ngram"))
                cand.add(sentence)
                if any(t.pos_ == "PROPN" or t.ent_type_ in _NAME_ENTS for t in span):
                    cand.names += 1

    frequent = {k: c for k, c in found.items() if c.frequency >= MIN_NGRAM_FREQ and c.names == 0}
    subsumed: set[str] = set()
    for text, cand in frequent.items():
        words = text.split()
        if len(words) < 3:
            continue
        for sub in (" ".join(words[:-1]), " ".join(words[1:])):
            if sub in frequent and frequent[sub].frequency <= cand.frequency:
                subsumed.add(sub)
    return {k: c for k, c in frequent.items() if k not in subsumed}


def collect_candidates(
    sentences: Sequence[Sentence], nlp: Language | None = None
) -> list[Candidate]:
    """Phrasal verbs + n-gramas, con presupuestos separados (tope ``MAX_CANDIDATES``).

    Primero los ``MAX_PHRASAL_CANDIDATES`` phrasal verbs más frecuentes; el resto del cupo
    se llena con los n-gramas más frecuentes. Si un texto sale por ambos caminos cuenta
    como phrasal (con la frecuencia mayor).
    """
    nlp = nlp or load_nlp_parser()
    docs = list(zip(sentences, nlp.pipe((s.text for s in sentences), batch_size=256), strict=True))
    ngrams = ngram_candidates(docs, nlp.Defaults.stop_words)
    phrasals = phrasal_candidates(docs)
    for text, cand in phrasals.items():
        other = ngrams.pop(text, None)
        if other is not None and other.frequency > cand.frequency:
            phrasals[text] = Candidate(
                text, "phrasal", other.frequency, other.example, other.example_score
            )

    def by_frequency(cands: Iterable[Candidate]) -> list[Candidate]:
        return sorted(cands, key=lambda c: (-c.frequency, c.text))

    chosen = by_frequency(phrasals.values())[: min(MAX_PHRASAL_CANDIDATES, MAX_CANDIDATES)]
    chosen += by_frequency(ngrams.values())[: MAX_CANDIDATES - len(chosen)]
    return chosen


# --- Filtro con OpenAI -----------------------------------------------------------------


class ExpressionVerdict(BaseModel):
    id: int = Field(description="Same id as the input item.")
    keep: bool
    expression: str | None
    kind: Literal["phrasal", "idiom"] | None


class ExpressionBatch(BaseModel):
    items: list[ExpressionVerdict]


class ExpressionCache:
    """``clave → ExpressionVerdict`` en un JSON. Clave: versión + modelo + candidato.

    No depende de la serie ni del ejemplo: un candidato ya juzgado no se vuelve a pagar
    aunque cambien ``--top``, ``--known-deck`` o la temporada.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        if path.is_file():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(text: str, model: str) -> str:
        raw = f"v{EXPR_FILTER_VERSION}|{model}|{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, model: str) -> ExpressionVerdict | None:
        entry = self._data.get(self.key(text, model))
        return ExpressionVerdict.model_validate(entry) if entry else None

    def put(self, text: str, model: str, verdict: ExpressionVerdict) -> None:
        self._data[self.key(text, model)] = verdict.model_dump()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._data)


def request_filter(
    client: Any, model: str, candidates: Sequence[Candidate], **retry_kwargs: Any
) -> dict[int, ExpressionVerdict]:
    """Juzga un lote de candidatos; devuelve ``{índice en el lote: veredicto}``."""
    items = [
        {"id": i, "text": c.text, "hint": c.hint, "sentence": c.example.text if c.example else ""}
        for i, c in enumerate(candidates)
    ]
    messages = [
        {"role": "system", "content": FILTER_PROMPT},
        {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
    ]
    parsed = parse_with_retries(
        client, model, messages, ExpressionBatch, error=ExpressionError, **retry_kwargs
    )
    return {v.id: _normalize(v) for v in parsed.items if 0 <= v.id < len(candidates)}


def _normalize(verdict: ExpressionVerdict) -> ExpressionVerdict:
    expression = " ".join((verdict.expression or "").lower().split()) or None
    keep = verdict.keep and expression is not None and len(expression.split()) >= 2
    return verdict.model_copy(
        update={
            "keep": keep,
            "expression": expression if keep else None,
            "kind": (verdict.kind or "idiom") if keep else None,
        }
    )


def filter_candidates(
    candidates: Sequence[Candidate],
    *,
    cache: ExpressionCache,
    model: str,
    client_factory: Callable[[], Any],
    batch_size: int = EXPR_BATCH_SIZE,
    on_batch: Callable[[int, int], None] | None = None,
    **retry_kwargs: Any,
) -> tuple[list[tuple[Candidate, ExpressionVerdict]], int]:
    """Veredictos de cada candidato (caché primero); devuelve ``(pares, nuevos)``.

    ``client_factory`` solo se llama si falta algo en el caché. Los candidatos que el
    modelo omita quedan fuera de esta corrida y se reintentan en la siguiente.
    """
    pending = [c for c in candidates if cache.get(c.text, model) is None]
    new = 0
    if pending:
        client = client_factory()
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            for i, verdict in request_filter(client, model, batch, **retry_kwargs).items():
                cache.put(batch[i].text, model, verdict)
                new += 1
            cache.save()
            if on_batch:
                on_batch(min(start + batch_size, len(pending)), len(pending))
    pairs = [(c, v) for c in candidates if (v := cache.get(c.text, model)) is not None]
    return pairs, new


# --- Ubicar la expresión en una oración (validación y resaltado) -----------------------


def _matches(token, word: str) -> bool:
    return token.lemma_.lower() == word or token.text.lower() == word


def find_expression(doc: Doc, expression: str) -> list[int] | None:
    """Índices de los tokens de ``expression`` en ``doc`` (solo los de la expresión).

    Primero busca la secuencia contigua de lemas ("makes sense", "get rid of"). Si no, una
    versión separable que empieza en un verbo, con hasta ``MAX_GAP`` tokens sin puntuación
    entre partes: "figure it out" → [figure, out] (el pronombre no se marca).
    """
    words = expression.lower().split()
    n = len(words)
    if not n:
        return None
    for start in range(len(doc) - n + 1):
        if all(_matches(doc[start + k], w) for k, w in enumerate(words)):
            return list(range(start, start + n))
    for start, token in enumerate(doc):
        if token.pos_ not in {"VERB", "AUX"} or not _matches(token, words[0]):
            continue
        found, pos = [start], start + 1
        for word in words[1:]:
            hit = None
            for k in range(pos, min(pos + MAX_GAP + 1, len(doc))):
                if _matches(doc[k], word):
                    hit = k
                    break
                if doc[k].is_punct:
                    break
            if hit is None:
                break
            found.append(hit)
            pos = hit + 1
        if len(found) == n:
            return found
    return None


# --- Selección final -------------------------------------------------------------------


def select_expressions(
    pairs: Sequence[tuple[Candidate, ExpressionVerdict]],
    series: str,
    *,
    limit: int,
    known: set[str] | None = None,
    nlp: Language | None = None,
) -> list[Card]:
    """Tarjetas de las ``limit`` expresiones más frecuentes que no estén en ``known``.

    Candidatos que normalizan igual se unen (frecuencia máxima, mejor ejemplo). El
    ejemplo debe contener la forma normalizada (``find_expression``); si ningún ejemplo
    la contiene, la expresión se descarta (mejor sin tarjeta que con un ejemplo engañoso).
    """
    known = known or set()
    groups: dict[tuple[str, str], list[Candidate]] = {}
    for cand, verdict in pairs:
        if verdict.keep and verdict.expression and verdict.expression not in known:
            groups.setdefault((verdict.expression, verdict.kind or "idiom"), []).append(cand)
    if not groups or limit <= 0:
        return []

    nlp = nlp or load_nlp()
    examples = sorted(
        {c.example.text for cands in groups.values() for c in cands if c.example is not None}
    )
    docs = dict(zip(examples, nlp.pipe(examples, batch_size=256), strict=True))

    chosen: list[tuple[int, str, str, Sentence]] = []
    for (expression, kind), cands in groups.items():
        valid = [
            c
            for c in cands
            if c.example is not None and find_expression(docs[c.example.text], expression)
        ]
        if not valid:
            continue
        best = min(valid, key=lambda c: c.example_score)
        frequency = max(c.frequency for c in cands)
        chosen.append((frequency, expression, kind, best.example))

    chosen = _drop_preposition_variants(chosen)
    chosen.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        Card(
            series=series,
            lemma=expression,
            pos=kind,
            rank=rank,
            frequency=frequency,
            example_en=example.text,
            source=example.source,
        )
        for rank, (frequency, expression, kind, example) in enumerate(chosen[:limit], start=1)
    ]


def _drop_preposition_variants(
    chosen: list[tuple[int, str, str, Sentence]],
) -> list[tuple[int, str, str, Sentence]]:
    """Si hay "hook up" (phrasal de 2 palabras) y "hook up with", deja solo una.

    Gana la más frecuente; en empate, la base de 2 palabras. Solo aplica cuando la variante
    añade una preposición de ``_VARIANT_PREPS`` al final.
    """
    by_text = {item[1]: item for item in chosen}
    dropped: set[str] = set()
    for frequency, expression, _kind, _example in chosen:
        words = expression.split()
        if len(words) != 3 or words[-1] not in _VARIANT_PREPS:
            continue
        base = by_text.get(" ".join(words[:2]))
        if base is None or base[2] != "phrasal":
            continue
        dropped.add(expression if base[0] >= frequency else base[1])
    return [item for item in chosen if item[1] not in dropped]


def merge_ranked(*groups: Sequence[Card]) -> list[Card]:
    """Une tarjetas de palabras y expresiones por frecuencia y re-numera ``rank``."""
    cards = sorted(
        (c for group in groups for c in group), key=lambda c: (-c.frequency, c.lemma, c.pos)
    )
    return [c.model_copy(update={"rank": i}) for i, c in enumerate(cards, start=1)]
