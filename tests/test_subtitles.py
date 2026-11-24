from pathlib import Path

import pytest

from anki_series.subtitles import (
    clean_line,
    cues_to_sentences,
    decode_bytes,
    find_srt_files,
    is_credit_line,
    load_sentences,
    parse_srt,
    split_sentences,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample.srt"


def srt(*cues: str) -> str:
    """Arma un .srt mínimo: cada argumento es el texto de un cue (1 s cada uno)."""
    blocks = []
    for i, text in enumerate(cues, start=1):
        blocks.append(f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},900\n{text}\n")
    return "\n".join(blocks)


def texts(content: str) -> list[str]:
    return [s.text for s in cues_to_sentences(parse_srt(content), "x.srt")]


# --- Fixture completo ------------------------------------------------------------------


def test_sample_fixture_end_to_end():
    result = [(s.source.start, s.text) for s in load_sentences(FIXTURE)]
    assert result == [
        ("00:00:01,000", "Previously on The Night Shift..."),
        ("00:00:07,200", "We need to bail before the cops show up."),
        ("00:00:12,800", "Relax, man."),
        ("00:00:12,800", "Nobody's gonna find us here."),
        ("00:00:16,000", "Did you grab the keys?"),
        ("00:00:16,000", "They're in the car."),
        ("00:00:19,700", "I can't believe you left the damn keys in the car."),
        ("00:00:23,400", "Running late again, huh?"),
        ("00:00:26,600", "I was running errands."),
        ("00:00:26,600", "The bank was packed."),
        ("00:00:30,200", "We're gonna run out of time if we don't leave right now."),
    ]


def test_sample_fixture_source_file_is_name():
    assert {s.source.file for s in load_sentences(FIXTURE)} == {"sample.srt"}


# --- Encoding --------------------------------------------------------------------------


def test_decode_utf8_with_bom():
    assert decode_bytes(b"\xef\xbb\xbfHello") == "Hello"


def test_decode_cp1252():
    text = srt("I don’t know, Zoë. That café’s closed.", "“Señor” is what they call him.")
    assert decode_bytes(text.encode("cp1252")) == text


def test_load_sentences_from_cp1252_file(tmp_path: Path):
    path = tmp_path / "ep.srt"
    path.write_bytes(srt("That café’s closed.").encode("cp1252"))
    assert [s.text for s in load_sentences(path)] == ["That café's closed."]


# --- Parseo ----------------------------------------------------------------------------


def test_parse_handles_crlf_and_normalizes_timestamps():
    content = "1\r\n0:00:01.5 --> 0:00:02.000\r\nHello there.\r\n\r\n"
    [cue] = parse_srt(content)
    assert cue.start == "00:00:01,500"
    assert cue.end == "00:00:02,000"
    assert cue.lines == ("Hello there.",)


def test_parse_skips_blocks_without_timing():
    assert len(parse_srt("garbage\n\n" + srt("Hi."))) == 1


# --- Limpieza --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<i>Hello</i> there", "Hello there"),
        ('<font color="#fff">Hi</font>', "Hi"),
        ("{\\an8}Up here.", "Up here."),
        ("[laughs] That's funny.", "That's funny."),
        ("(door slams)", ""),
        ("♪ La la la ♪", ""),
        ("MARCUS: Let's go.", "Let's go."),
        ("MAN #2: Over here!", "Over here!"),
        ("- Did you?", "- Did you?"),
        ("- JOE: Yeah.", "- Yeah."),
        ("I don’t “care”.", 'I don\'t "care".'),
        ("...", ""),
    ],
)
def test_clean_line(raw: str, expected: str):
    assert clean_line(raw) == expected


def test_keeps_slang_and_profanity():
    assert texts(srt("Shit, that's a damn good burger, y'all.")) == [
        "Shit, that's a damn good burger, y'all."
    ]


# --- Segmentación y unión --------------------------------------------------------------


def test_split_respects_abbreviations():
    assert split_sentences("Mr. Smith is here. Dr. Who left!") == [
        "Mr. Smith is here.",
        "Dr. Who left!",
    ]


def test_joins_sentence_split_across_cues_with_first_timestamp():
    sentences = cues_to_sentences(parse_srt(srt("I was going", "to call you.")), "x.srt")
    assert [s.text for s in sentences] == ["I was going to call you."]
    assert sentences[0].source.start == "00:00:01,000"


def test_ellipsis_continuation_is_joined():
    assert texts(srt("I was going to...", "...call you.")) == ["I was going to call you."]


def test_ellipsis_followed_by_new_sentence_is_kept_apart():
    assert texts(srt("I was going to...", "Forget it.")) == [
        "I was going to...",
        "Forget it.",
    ]


def test_dialogue_dash_never_joins_previous_text():
    assert texts(srt("- Are you", "- No.")) == ["Are you", "No."]


def test_sound_only_cue_does_not_break_a_sentence():
    assert texts(srt("We should", "[thunder]", "leave now.")) == ["We should leave now."]


def test_unterminated_text_at_end_of_file_is_emitted():
    assert texts(srt("Wait for me")) == ["Wait for me"]


# --- Créditos de subtítulos ------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "== sync, corrected by elderman ==",
        "<font color=#00FF00>== sync, corrected by elderman ==</font>",
        "Synced and corrected by honeybunny",
        "Subtitles by DramaFever",
        "Resync by someone, corrected by another",
        "Downloaded from www.OpenSubtitles.org",
        "Best watched using Open Subtitles MKV Player",
        "www.addic7ed.com",
        "Support us at tvsubtitles.net",
        "{\\an8}== @elder_man ==",
    ],
)
def test_credit_lines_are_removed(line: str):
    assert clean_line(line) == ""


def test_credit_cue_is_dropped_without_breaking_dialogue():
    content = srt(
        "We need to",
        "== sync, corrected by\nelderman ==",  # crédito partido en dos líneas
        "settle this case.",
        "Synced and corrected by honeybunny\nwww.addic7ed.com",
    )
    assert texts(content) == ["We need to settle this case."]


@pytest.mark.parametrize(
    "line",
    [
        "I corrected the file myself.",
        "We need to sync our calendars.",
        "She was the subject of the case.",
        "Go to the bank, Mike.",
    ],
)
def test_dialogue_is_not_mistaken_for_credits(line: str):
    assert not is_credit_line(line)
    assert clean_line(line) == line


# --- Archivos y carpetas ---------------------------------------------------------------


def test_folder_is_read_recursively_in_order(tmp_path: Path):
    (tmp_path / "S01").mkdir()
    (tmp_path / "S01" / "E02.srt").write_text(srt("Second."), encoding="utf-8")
    (tmp_path / "S01" / "E01.srt").write_text(srt("First."), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    sentences = load_sentences(tmp_path)
    assert [(s.source.file, s.text) for s in sentences] == [
        ("S01/E01.srt", "First."),
        ("S01/E02.srt", "Second."),
    ]


def test_empty_folder_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="no contiene"):
        find_srt_files(tmp_path)


def test_non_srt_file_raises(tmp_path: Path):
    path = tmp_path / "x.txt"
    path.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="No es un archivo .srt"):
        find_srt_files(path)


def test_missing_path_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_srt_files(tmp_path / "nope.srt")
