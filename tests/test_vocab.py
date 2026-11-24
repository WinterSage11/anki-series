from pathlib import Path

import pytest
from apkg_factory import make_apkg

from anki_series.models import Sentence, SourceRef
from anki_series.vocab import (
    KnownDeckError,
    build_vocabulary,
    count_lemmas,
    known_lemmas_from_words,
    load_known_lemmas,
    read_known_words,
    strip_html,
)


def sents(*texts: str) -> list[Sentence]:
    return [
        Sentence(text=t, source=SourceRef(file="x.srt", start=f"00:00:{i:02d},000"))
        for i, t in enumerate(texts)
    ]


def keys(cards) -> list[tuple[str, str]]:
    return [(c.lemma, c.pos) for c in cards]


# --- Conteo y filtros ------------------------------------------------------------------


def test_lemma_and_pos_are_separate_cards():
    cards = build_vocabulary(
        sents("I run every morning before work.", "That was a long run.", "She runs fast."),
        "S",
    )
    found = {(c.lemma, c.pos): c.frequency for c in cards}
    assert found[("run", "VERB")] == 2  # run + runs → mismo lema
    assert found[("run", "NOUN")] == 1


def test_filters_names_stopwords_numbers_and_punctuation():
    cards = build_vocabulary(sents("Marcus said this shit is crazy.", "We paid 200 bucks!"), "S")
    lemmas = {c.lemma for c in cards}
    assert lemmas == {"say", "shit", "crazy", "pay", "buck"}  # slang y groserías se quedan
    assert "marcus" not in lemmas and "200" not in lemmas and "this" not in lemmas


def test_person_entities_are_excluded():
    cards = build_vocabulary(sents("Run, Sarah, run!"), "S")
    assert "sarah" not in {c.lemma for c in cards}


def test_content_words_in_spacy_stoplist_are_kept():
    # "get", "go" y "front" están en la stoplist de spaCy, pero son vocabulario útil.
    cards = build_vocabulary(sents("Go get the car.", "It is really in front."), "S")
    lemmas = {c.lemma for c in cards}
    assert {"go", "get", "front"} <= lemmas
    assert "really" not in lemmas  # adverbio de la stoplist: fuera


def test_ranking_by_frequency_then_alphabetical_and_top():
    cards = build_vocabulary(sents("Okay, dude.", "Okay, dude, damn.", "Okay."), "S", top=2)
    assert keys(cards) == [("okay", "INTJ"), ("dude", "NOUN")]
    assert [c.rank for c in cards] == [1, 2]
    assert [c.frequency for c in cards] == [3, 2]


def test_example_prefers_sentence_close_to_ideal_length():
    short = "Keys!"
    good = "I can't believe you left the keys in the car again."
    cards = build_vocabulary(sents(short, good, "Where are my keys?"), "S")
    key = next(c for c in cards if c.lemma == "key")
    assert key.example_en == good
    assert key.source.start == "00:00:01,000"


def test_example_tie_keeps_first_occurrence():
    candidates = count_lemmas(sents("Grab the keys now, please.", "Grab the bags now, please."))
    assert candidates[("grab", "VERB")].example.text == "Grab the keys now, please."


def test_known_lemmas_are_excluded_for_every_pos():
    texts = ("I run every morning before work.", "That was a long run.")
    cards = build_vocabulary(sents(*texts), "S", known_lemmas={"run"})
    assert "run" not in {c.lemma for c in cards}
    assert ("work", "NOUN") in keys(cards)


def test_exclusion_happens_before_top_cut():
    cards = build_vocabulary(
        sents("Damn, dude.", "Damn it.", "Okay."), "S", top=1, known_lemmas={"damn"}
    )
    assert keys(cards) == [("dude", "NOUN")]


def test_cards_carry_series():
    cards = build_vocabulary(sents("Relax, dude."), "The Night Shift")
    assert all(c.series == "The Night Shift" for c in cards)


# --- Mazo conocido ---------------------------------------------------------------------


def test_strip_html():
    assert strip_html("<b>store</b>&nbsp;") == "store"
    assert strip_html("left [sound:left.mp3]") == "left"
    assert strip_html("<div>rock&amp;roll</div>") == "rock&roll"


@pytest.mark.parametrize("fmt", ["anki21", "anki21b", "anki2"])
def test_read_known_words_formats(tmp_path: Path, fmt: str):
    words = read_known_words(make_apkg(tmp_path, fmt), 1)
    assert words == ["have", "store", "left", "give up"]


def test_read_known_words_by_field_name(tmp_path: Path):
    apkg = make_apkg(tmp_path, "anki21b")
    assert read_known_words(apkg, "palabra") == read_known_words(apkg, "1")


def test_numeric_field_raises_with_available_fields(tmp_path: Path):
    with pytest.raises(KnownDeckError, match="'Palabra'"):
        read_known_words(make_apkg(tmp_path), 0)


def test_invalid_apkg(tmp_path: Path):
    bad = tmp_path / "bad.apkg"
    bad.write_text("not a zip")
    with pytest.raises(KnownDeckError):
        read_known_words(bad, 1)
    with pytest.raises(FileNotFoundError):
        read_known_words(tmp_path / "missing.apkg", 1)


def test_known_lemmas_ignore_multiword_and_include_lemma():
    lemmas = known_lemmas_from_words(["Have", "left", "give up"])
    assert {"have", "left", "leave"} <= lemmas
    assert "give" not in lemmas and "up" not in lemmas


def test_load_known_lemmas_end_to_end(tmp_path: Path):
    lemmas = load_known_lemmas(make_apkg(tmp_path), "Palabra")
    assert {"have", "store", "leave"} <= lemmas
