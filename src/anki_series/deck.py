"""Fase 3: mazo de Anki (.apkg) con genanki a partir de ``Card``.

La plantilla ya trae todos los campos (traducción, imagen, créditos) aunque en esta fase
lleguen vacíos: si el modelo de nota cambiara de campos después, Anki no podría actualizar
las notas ya importadas. Por eso ``MODEL_ID`` y ``DECK_ID`` son constantes fijas y el GUID
de cada nota es determinista: reimportar el mazo actualiza las notas en vez de duplicarlas.
"""

import hashlib
import html
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import genanki
from spacy.language import Language

from anki_series.models import Card, ImageStatus
from anki_series.vocab import load_nlp

# Generado una sola vez con random.randrange(1 << 30, 1 << 31). NUNCA cambiarlo: es global
# para todas las series. El deck_id, en cambio, se deriva de la serie (ver ``deck_id``).
MODEL_ID = 2104579187

MODEL_NAME = "anki-series (EN → ES)"
# El orden y los nombres de los campos son parte del esquema: no cambiarlos después de
# publicar un mazo. AudioWord/AudioExample guardan ``[sound:archivo.mp3]`` (vacíos hasta
# tener voces neuronales). Credits y Source son metadata: ninguna plantilla los muestra.
FIELDS = [
    "Word",
    "POS",
    "ExampleEN",
    "Translation",
    "ExampleES",
    "Image",
    "AudioWord",
    "AudioExample",
    "Credits",
    "Source",
]

FRONT = """\
<div class="box">
  <section class="front">
    {{#AudioWord}}<div class="audio corner">{{AudioWord}}</div>{{/AudioWord}}
    <div class="word">{{Word}}</div>
    <div class="pos">{{POS}}</div>
  </section>
</div>
"""

BACK = """\
<div class="box">
  <section class="head">
    <span class="word">{{Word}}</span>
    {{#AudioWord}}<span class="audio">{{AudioWord}}</span>{{/AudioWord}}
    <span class="pos">{{POS}}</span>
  </section>
  {{#Translation}}
  <section>
    <div class="label">traducción:</div>
    <div class="translation">{{Translation}}</div>
  </section>
  {{/Translation}}
  {{#Image}}<section class="image">{{Image}}</section>{{/Image}}
  <section>
    <div class="label">ejemplo:</div>
    <div class="row">
      <div class="example">{{ExampleEN}}</div>
      {{#AudioExample}}<div class="audio">{{AudioExample}}</div>{{/AudioExample}}
    </div>
    {{#ExampleES}}<div class="example-es">{{ExampleES}}</div>{{/ExampleES}}
  </section>
</div>
"""

CSS = """\
.card {
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, sans-serif;
  font-size: 18px; color: #1d1d1f; background: #f4f4f5; text-align: left; margin: 0;
}
.box {
  max-width: 400px; margin: 20px auto; background: #fff; border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08); overflow: hidden;
}
.box section { padding: 14px 18px; border-top: 1px solid #e8e8ea; }
.box section:first-child { border-top: none; }

.front { position: relative; text-align: center; padding: 56px 18px 48px !important; }
.front .word { font-size: 28px; }
.corner { position: absolute; top: 12px; right: 14px; }

.head { display: flex; align-items: center; gap: 12px; padding: 26px 18px !important; }
.word { font-size: 26px; font-weight: 700; }
.pos {
  font-size: 11px; letter-spacing: 0.06em; color: #9a9aa0; text-transform: lowercase;
}
.head .pos { margin-left: auto; }

.label { font-size: 14px; color: #9a9aa0; margin-bottom: 6px; }
.translation { font-size: 18px; padding-left: 12px; }
.image { text-align: center; }
.image img { max-width: 100%; max-height: 200px; }
.row { display: flex; align-items: center; gap: 12px; padding-left: 12px; }
.example { flex: 1; }
.example b { color: #6d4aff; }
.example-es { font-style: italic; color: #6e6e73; padding-left: 12px; margin-top: 4px; }

.audio .replay-button svg { width: 26px; height: 26px; }

.nightMode.card, .nightMode .card { background: #1c1c1e; color: #f2f2f7; }
.nightMode .box { background: #2c2c2e; box-shadow: none; }
.nightMode .box section { border-color: #3a3a3c; }
.nightMode .example b { color: #a48bff; }
.nightMode .example-es { color: #aeaeb2; }
"""

MODEL = genanki.Model(
    MODEL_ID,
    MODEL_NAME,
    fields=[{"name": name} for name in FIELDS],
    templates=[{"name": "EN → ES", "qfmt": FRONT, "afmt": BACK}],
    css=CSS,
)


def note_guid(card: Card) -> str:
    """GUID estable de la nota: depende solo de serie, lema y POS."""
    return genanki.guid_for(card.series, card.lemma, card.pos)


def deck_name(series: str) -> str:
    return f"anki-series::{series}"


def deck_id(series: str) -> int:
    """ID estable por serie (sha256 del nombre): series distintas no se mezclan al importar.

    Se mantiene en el rango que recomienda genanki, ``[2**30, 2**31)``.
    """
    digest = hashlib.sha256(f"anki-series|{series}".encode()).digest()
    return (1 << 30) + int.from_bytes(digest[:8], "big") % (1 << 30)


# --- Resaltado de la palabra en el ejemplo ---------------------------------------------


def highlight_examples(cards: Sequence[Card], nlp: Language | None = None) -> list[str]:
    """``example_en`` de cada tarjeta en HTML escapado, con la palabra en ``<b>``.

    La oración trae la forma flexionada ("runs", "bailed"), así que se vuelve a pasar por
    spaCy y se resaltan los tokens con el mismo lema y POS. Si no hay ninguno (spaCy puede
    etiquetar distinto fuera de contexto), se usa solo el lema; si tampoco, va sin resaltar.
    """
    nlp = nlp or load_nlp()
    docs = nlp.pipe((c.example_en for c in cards), batch_size=256)
    return [_highlight(doc, card) for card, doc in zip(cards, docs, strict=True)]


def _highlight(doc, card: Card) -> str:
    same_lemma = {t.i for t in doc if t.lemma_.lower() == card.lemma}
    exact = {i for i in same_lemma if doc[i].pos_ == card.pos}
    marked = exact or same_lemma
    parts = []
    for token in doc:
        text = html.escape(token.text)
        if token.i in marked:
            text = f"<b>{text}</b>"
        parts.append(text + token.whitespace_)
    return "".join(parts)


# --- Construcción del mazo -------------------------------------------------------------


def _media_name(card: Card, path: Path) -> str:
    """Nombre único en la colección de Anki (la carpeta de media es global)."""
    digest = hashlib.sha1(card.key.encode()).hexdigest()[:12]
    return f"anki-series-{digest}{path.suffix.lower() or '.jpg'}"


def _image_path(card: Card) -> Path | None:
    if card.image.status != ImageStatus.FOUND or not card.image.path:
        return None
    path = Path(card.image.path)
    if not path.is_file():
        raise FileNotFoundError(f"Imagen de {card.lemma}/{card.pos} no encontrada: {path}")
    return path


def _credits(card: Card) -> str:
    info = card.image
    parts = [p for p in (info.credit, info.source, info.source_url) if p]
    return html.escape(" · ".join(parts))


def _fields(card: Card, example_html: str, image_html: str) -> list[str]:
    return [
        html.escape(card.lemma),
        card.pos,
        example_html,
        html.escape(card.translation or ""),
        html.escape(card.example_es or ""),
        image_html,
        "",  # AudioWord: [sound:...] cuando haya voces neuronales
        "",  # AudioExample
        _credits(card) if image_html else "",
        html.escape(f"{card.source.file} {card.source.start}"),
    ]


def build_deck(cards: Sequence[Card], output: Path, nlp: Language | None = None) -> Path:
    """Escribe ``output`` (.apkg) con una nota por tarjeta y las imágenes encontradas."""
    if not cards:
        raise ValueError("No hay tarjetas para construir el mazo.")
    series = {c.series for c in cards}
    if len(series) != 1:
        raise ValueError(f"Todas las tarjetas deben ser de una sola serie: {sorted(series)}")

    series_name = series.pop()
    deck = genanki.Deck(deck_id(series_name), deck_name(series_name))
    examples = highlight_examples(cards, nlp)
    media: dict[str, Path] = {}
    for card, example_html in zip(cards, examples, strict=True):
        image_html = ""
        if (path := _image_path(card)) is not None:
            name = _media_name(card, path)
            media[name] = path
            image_html = f'<img src="{name}">'
        deck.add_note(
            genanki.Note(
                model=MODEL,
                fields=_fields(card, example_html, image_html),
                guid=note_guid(card),
                tags=[card.pos],
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    # genanki empaqueta cada media con su nombre de archivo: se copian con el nombre único.
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for name, src in media.items():
            dest = Path(tmp) / name
            shutil.copyfile(src, dest)
            files.append(str(dest))
        genanki.Package(deck, media_files=files).write_to_file(str(output))
    return output
