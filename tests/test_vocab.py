from pathlib import Path

import pytest
from apkg_factory import make_apkg

from anki_series.models import Sentence, SourceRef
from anki_series.vocab import (
    KnownDeckError,
    build_vocabulary,
    common_words,
    count_lemmas,
    is_filler,
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
    texts = ("The dude has a car.", "That dude lost the car.", "A dude, a car, a gun.")
    cards = build_vocabulary(sents(*texts, "Hide the gun."), "S", top=2)
    assert keys(cards) == [("car", "NOUN"), ("dude", "NOUN")]  # empate 3-3 → alfabético
    assert [c.rank for c in cards] == [1, 2]
    assert [c.frequency for c in cards] == [3, 3]


def test_interjections_are_excluded():
    cards = build_vocabulary(sents("Yeah, okay.", "Oh, hey!", "Uh, yes.", "Wow, nice car."), "S")
    assert all(c.pos != "INTJ" for c in cards)
    assert not {"yeah", "okay", "oh", "hey", "uh", "yes", "wow"} & {c.lemma for c in cards}
    assert ("car", "NOUN") in keys(cards)


@pytest.mark.parametrize(
    "text",
    [
        "Mm-hmm. I know the file.",
        "Mm, good file.",
        "Mhm, sure.",
        "Hmmm, the file.",
        "Uhh, um, file.",
    ],
)
def test_vocal_fillers_are_excluded(text: str):
    # spaCy etiqueta "Mm-hmm" como mm/NOUN + hmm/NOUN y "Mm" como ADJ.
    lemmas = {lemma for lemma, _pos in count_lemmas(sents(text))}
    assert not lemmas & {"hmm", "mm", "mhm", "hmmm", "uhh", "um"}
    assert "file" in lemmas or text.startswith("Mhm")


@pytest.mark.parametrize("word", ["hmm", "mm", "mhm", "uh", "um", "huh", "ah", "er", "hmmm", "ahh"])
def test_is_filler(word: str):
    assert is_filler(word)


@pytest.mark.parametrize("word", ["err", "hum", "aha", "mom", "man", "uhm-no", "hat"])
def test_real_words_are_not_fillers(word: str):
    assert not is_filler(word)


def test_character_names_tagged_as_noun_are_excluded():
    # spaCy etiqueta "Harvey" como NOUN en vocativo y como PROPN en el resto.
    texts = ("Harvey, you gotta see this.", "I gotta tell Harvey.", "Harvey Specter is here.")
    candidates = count_lemmas(sents(*texts, "Call Harvey now."))
    assert not any(lemma == "harvey" for lemma, _pos in candidates)
    assert ("see", "VERB") in candidates


def test_occasional_propn_tag_does_not_drop_common_word():
    texts = ("The car is red.", "I sold the car.", "Get in the car.", "Car keys, now.")
    assert ("car", "NOUN") in count_lemmas(sents(*texts))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Gonna need a bigger boat.", ("go", "VERB")),  # spaCy: "Gon"/go/NOUN
        ("I'm gonna kill him.", ("go", "VERB")),
        ("We gotta leave now.", ("get", "VERB")),
        ("You wanna dance?", ("want", "VERB")),
    ],
)
def test_informal_contractions_map_to_verb(text: str, expected: tuple[str, str]):
    candidates = count_lemmas(sents(text))
    assert expected in candidates
    assert ("go", "NOUN") not in candidates and not any(k[0] == "wanna" for k in candidates)


def test_split_tokens_without_contraction_are_untouched():
    # "got to" con espacio no es "gotta": se queda como get/VERB normal.
    assert ("get", "VERB") in count_lemmas(sents("I got to the store late."))


def test_common_words_are_only_words():
    words = common_words(100)
    assert len(words) == 100
    assert {"the", "you", "have"} <= words
    assert all(w.isalpha() for w in words)  # sin números ni contracciones ("don't")
    assert common_words(0) == set()


def test_exclude_common_words_before_top_cut():
    texts = ("Take the briefcase.", "Take the briefcase now.", "Settle the case.")
    cards = build_vocabulary(sents(*texts), "S", top=1, known_lemmas=common_words(500))
    assert keys(cards) == [("briefcase", "NOUN")]


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
