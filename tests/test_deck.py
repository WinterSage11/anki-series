import json
import sqlite3
import zipfile
from pathlib import Path

import genanki
import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from anki_series.cli import app
from anki_series.deck import (
    BACK,
    FIELDS,
    FRONT,
    MODEL_ID,
    build_deck,
    deck_id,
    highlight_examples,
    note_guid,
)
from anki_series.models import Card, ImageInfo, ImageStatus, SourceRef
from anki_series.vocab import read_known_words

runner = CliRunner()


def card(lemma: str, pos: str, example: str, **kw) -> Card:
    defaults = dict(
        series="The Night Shift",
        rank=1,
        frequency=3,
        source=SourceRef(file="S01E01.srt", start="00:01:02,300"),
    )
    return Card(lemma=lemma, pos=pos, example_en=example, **(defaults | kw))


def read_apkg(apkg: Path, tmp_path: Path) -> dict:
    """Abre el .apkg como lo haría Anki: zip → collection.anki2 (SQLite) + mapa de media."""
    with zipfile.ZipFile(apkg) as zf:
        assert zf.testzip() is None
        media = json.loads(zf.read("media"))
        files = {name: zf.read(idx) for idx, name in media.items()}
        db_path = tmp_path / f"{apkg.stem}.anki2"
        db_path.write_bytes(zf.read("collection.anki2"))
    db = sqlite3.connect(db_path)
    notes = {
        guid: dict(zip(FIELDS, flds.split("\x1f"), strict=True)) | {"tags": tags.split()}
        for guid, flds, tags in db.execute("SELECT guid, flds, tags FROM notes")
    }
    models, decks = (json.loads(v) for v in db.execute("SELECT models, decks FROM col").fetchone())
    n_cards = db.execute("SELECT count(*) FROM cards").fetchone()[0]
    db.close()
    return dict(notes=notes, models=models, decks=decks, media=files, n_cards=n_cards)


@pytest.fixture
def cards() -> list[Card]:
    return [
        card("bail", "VERB", "We need to bail before the cops show up."),
        card("run", "NOUN", "That was a long run, & I'm tired.", rank=2),
    ]


# --- .apkg válido ----------------------------------------------------------------------


def test_build_writes_valid_apkg(tmp_path: Path, cards: list[Card]):
    apkg = build_deck(cards, tmp_path / "out" / "deck.apkg")
    data = read_apkg(apkg, tmp_path)

    assert len(data["notes"]) == 2 and data["n_cards"] == 2
    assert str(MODEL_ID) in data["models"]
    assert [f["name"] for f in data["models"][str(MODEL_ID)]["flds"]] == FIELDS
    assert data["decks"][str(deck_id("The Night Shift"))]["name"] == "anki-series::The Night Shift"

    note = data["notes"][note_guid(cards[0])]
    assert note["Word"] == "bail" and note["POS"] == "VERB"
    for empty in ("Translation", "ExampleES", "Image", "AudioWord", "AudioExample"):
        assert note[empty] == ""
    assert note["Source"] == "S01E01.srt 00:01:02,300"
    assert note["tags"] == ["VERB"]
    # El lector del mazo conocido (F2) también entiende el .apkg generado.
    assert sorted(read_known_words(apkg, "Word")) == ["bail", "run"]


def test_example_highlights_inflected_form_and_escapes_html(tmp_path: Path, cards: list[Card]):
    cards.append(card("cop", "NOUN", "The cops are here."))
    examples = highlight_examples(cards)
    assert examples[0] == "We need to <b>bail</b> before the cops show up."
    assert examples[1] == "That was a long <b>run</b>, &amp; I&#x27;m tired."
    assert examples[2] == "The <b>cops</b> are here."  # forma flexionada


def test_example_without_match_is_left_plain():
    [example] = highlight_examples([card("bail", "VERB", "Nothing to see here.")])
    assert "<b>" not in example


def test_translation_fields_are_filled_when_present(tmp_path: Path):
    c = card(
        "bail", "VERB", "We need to bail.", translation="pelarse", example_es="Hay que pelarnos."
    )
    note = read_apkg(build_deck([c], tmp_path / "d.apkg"), tmp_path)["notes"][note_guid(c)]
    assert note["Translation"] == "pelarse"
    assert note["ExampleES"] == "Hay que pelarnos."


# --- GUID estable ----------------------------------------------------------------------


def test_guid_is_deterministic_from_series_lemma_pos(cards: list[Card]):
    c = cards[0]
    assert note_guid(c) == genanki.guid_for("The Night Shift", "bail", "VERB")
    assert note_guid(c) == note_guid(c.model_copy(update={"rank": 99, "example_en": "Bail!"}))
    assert note_guid(c) != note_guid(c.model_copy(update={"pos": "NOUN"}))
    assert note_guid(c) != note_guid(c.model_copy(update={"series": "Other"}))


def test_rebuild_keeps_guids_and_ids(tmp_path: Path, cards: list[Card]):
    first = read_apkg(build_deck(cards, tmp_path / "a.apkg"), tmp_path)
    second = read_apkg(build_deck(cards[::-1], tmp_path / "b.apkg"), tmp_path)
    assert first["notes"].keys() == second["notes"].keys()
    assert first["models"].keys() == second["models"].keys() == {str(MODEL_ID)}


# --- Con y sin imagen ------------------------------------------------------------------


def test_cards_with_and_without_image(tmp_path: Path, cards: list[Card]):
    img = tmp_path / "dog.JPG"
    img.write_bytes(b"\xff\xd8fake-jpeg")
    cards[0].image = ImageInfo(
        status=ImageStatus.FOUND,
        path=str(img),
        source="pixabay",
        source_url="https://pixabay.com/photos/1",
        credit="Jane",
    )
    cards[1].image = ImageInfo(status=ImageStatus.SKIPPED, reason="abstracta")

    data = read_apkg(build_deck(cards, tmp_path / "d.apkg"), tmp_path)
    with_img = data["notes"][note_guid(cards[0])]
    without = data["notes"][note_guid(cards[1])]

    [(name, content)] = data["media"].items()
    assert name.startswith("anki-series-") and name.endswith(".jpg")
    assert content == b"\xff\xd8fake-jpeg"
    assert with_img["Image"] == f'<img src="{name}">'
    assert with_img["Credits"] == "Jane · pixabay · https://pixabay.com/photos/1"
    assert without["Image"] == without["Credits"] == ""


def test_found_image_missing_on_disk_fails(tmp_path: Path, cards: list[Card]):
    cards[0].image = ImageInfo(status=ImageStatus.FOUND, path=str(tmp_path / "nope.jpg"))
    with pytest.raises(FileNotFoundError, match="bail/VERB"):
        build_deck(cards, tmp_path / "d.apkg")


# --- Plantilla -------------------------------------------------------------------------


def test_front_shows_only_word_pos_and_word_audio():
    assert "{{Word}}" in FRONT and "{{POS}}" in FRONT
    assert "{{#AudioWord}}" in FRONT and "{{/AudioWord}}" in FRONT
    for hidden in ("ExampleEN", "Image", "Translation", "ExampleES", "AudioExample"):
        assert hidden not in FRONT


def test_back_sections_in_order():
    order = [
        "{{Word}}",
        "{{#AudioWord}}",
        "traducción:",
        "{{Translation}}",
        "{{#Image}}",
        "ejemplo:",
        "{{ExampleEN}}",
        "{{#AudioExample}}",
        "{{ExampleES}}",
    ]
    positions = [BACK.index(part) for part in order]
    assert positions == sorted(positions)
    for field in ("Image", "AudioWord", "AudioExample", "Translation", "ExampleES"):
        assert f"{{{{#{field}}}}}" in BACK and f"{{{{/{field}}}}}" in BACK  # bloque condicional


def test_templates_have_no_tts_and_hide_metadata():
    assert "{{tts" not in FRONT + BACK
    assert "Credits" not in FRONT + BACK and "Source" not in FRONT + BACK


# --- deck_id por serie -----------------------------------------------------------------


def test_deck_id_is_deterministic_per_series():
    assert deck_id("The Night Shift") == deck_id("The Night Shift")
    assert deck_id("The Night Shift") != deck_id("Other Show")
    assert all((1 << 30) <= deck_id(s) < (1 << 31) for s in ("A", "B", "The Night Shift"))
    assert deck_id("The Night Shift") != MODEL_ID


def test_two_series_get_different_decks_same_model(tmp_path: Path, cards: list[Card]):
    other = [c.model_copy(update={"series": "Other Show"}) for c in cards]
    a = read_apkg(build_deck(cards, tmp_path / "a.apkg"), tmp_path)
    b = read_apkg(build_deck(other, tmp_path / "b.apkg"), tmp_path)
    assert str(deck_id("The Night Shift")) in a["decks"]
    assert str(deck_id("Other Show")) in b["decks"]
    assert a["models"].keys() == b["models"].keys() == {str(MODEL_ID)}
    assert not a["notes"].keys() & b["notes"].keys()


def test_mixed_series_or_empty_fails(tmp_path: Path, cards: list[Card]):
    with pytest.raises(ValueError):
        build_deck([], tmp_path / "d.apkg")
    cards[1].series = "Other"
    with pytest.raises(ValueError, match="una sola serie"):
        build_deck(cards, tmp_path / "d.apkg")


# --- CLI -------------------------------------------------------------------------------


def test_cli_build(tmp_path: Path, cards: list[Card]):
    words = tmp_path / "words.json"
    words.write_bytes(TypeAdapter(list[Card]).dump_json(cards))
    out = tmp_path / "deck.apkg"
    result = runner.invoke(app, ["build", "-i", str(words), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "Notas: 2 · Con traducción: 0 · Con imagen: 0" in result.output
    assert len(read_apkg(out, tmp_path)["notes"]) == 2


def test_cli_build_default_paths(tmp_path: Path, cards: list[Card], monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    out_dir = tmp_path / "data" / "output"
    out_dir.mkdir(parents=True)
    (out_dir / "words.json").write_bytes(TypeAdapter(list[Card]).dump_json(cards))
    result = runner.invoke(app, ["build"])
    assert result.exit_code == 0, result.output
    assert (out_dir / "the-night-shift.apkg").is_file()


def test_cli_build_missing_input_fails_cleanly(tmp_path: Path):
    result = runner.invoke(app, ["build", "-i", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "extract" in result.output
