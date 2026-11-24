"""Tests de la Fase 6. Cero llamadas reales: edge-tts se simula (sintetizador falso o
``edge_tts.Communicate`` reemplazado)."""

import asyncio
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest
from edge_tts.exceptions import NoAudioReceived, WebSocketError
from pydantic import TypeAdapter
from typer.testing import CliRunner

from anki_series import audio as audio_mod
from anki_series import cli
from anki_series.audio import (
    AudioCache,
    AudioError,
    edge_tts_synthesize,
    generate_audio,
    needs_work,
    plain_text,
    synthesize_with_retry,
)
from anki_series.deck import FIELDS, build_deck, note_guid
from anki_series.models import AudioInfo, Card, SourceRef

CARDS = TypeAdapter(list[Card])
runner = CliRunner()
VOICE = "en-US-AndrewNeural"


def make_card(lemma: str, pos: str = "NOUN", example: str | None = None) -> Card:
    return Card(
        series="X",
        lemma=lemma,
        pos=pos,
        rank=1,
        frequency=3,
        example_en=example if example is not None else f"I need the {lemma} right now.",
        source=SourceRef(file="e01.srt", start="00:00:01,000"),
    )


class FakeTTS:
    """Sintetizador falso: escribe ``mp3:<voz>:<texto>`` y registra cada llamada.

    ``fail[texto]`` es una lista de excepciones a lanzar (una por intento) antes de responder.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail: dict[str, list[BaseException]] = {}
        self.empty: set[str] = set()  # textos que dejan un archivo vacío
        self.active = 0
        self.max_active = 0

    async def __call__(self, text: str, voice: str, dest: Path) -> None:
        self.calls.append((text, voice))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            for _ in range(3):
                await asyncio.sleep(0)  # cede el turno para que corran otras tareas
            if self.fail.get(text):
                raise self.fail[text].pop(0)
            dest.write_bytes(b"" if text in self.empty else f"mp3:{voice}:{text}".encode())
        finally:
            self.active -= 1

    def texts(self) -> list[str]:
        return [t for t, _ in self.calls]


class NoSleep:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


@pytest.fixture
def cache(tmp_path: Path) -> AudioCache:
    return AudioCache(tmp_path / "cache" / "audio")


def run(cards: list[Card], tts: FakeTTS, cache: AudioCache, **kw):
    kw.setdefault("sleep", NoSleep())
    return generate_audio(cards, cache=cache, voice=kw.pop("voice", VOICE), synthesize=tts, **kw)


# --- Texto y caché ---------------------------------------------------------------------


def test_plain_text_strips_html_and_entities():
    assert plain_text("We need to <b>bail</b> &amp; run,  now.\n") == "We need to bail & run, now."
    assert plain_text("I'm <i>so</i> done") == "I'm so done"


def test_cache_key_depends_on_voice_and_text(cache: AudioCache):
    path = cache.path_for(VOICE, "run")
    assert path == cache.path_for(VOICE, "run") == cache.path_for(VOICE, "<b>run</b>")
    assert path.suffix == ".mp3" and path.parent == cache.dir
    assert path != cache.path_for("en-US-AriaNeural", "run")
    assert path != cache.path_for(VOICE, "running")


def test_old_cards_json_without_audio_loads_with_defaults():
    raw = json.loads(CARDS.dump_json([make_card("knife")]))
    del raw[0]["audio"]
    [card] = CARDS.validate_python(raw)
    assert card.audio == AudioInfo()


# --- Generación ------------------------------------------------------------------------


def test_generates_word_and_example_mp3(cache: AudioCache):
    cards = [make_card("bail", "VERB", "We need to <b>bail</b> &amp; go.")]
    tts = FakeTTS()
    [card], stats = run(cards, tts, cache)

    assert sorted(tts.calls) == [("We need to bail & go.", VOICE), ("bail", VOICE)]
    assert card.audio.voice == VOICE
    assert Path(card.audio.word).read_bytes() == f"mp3:{VOICE}:bail".encode()
    assert Path(card.audio.example).read_bytes() == f"mp3:{VOICE}:We need to bail & go.".encode()
    assert stats.generated == 2 and stats.from_cache == 0 and stats.keys == [card.key]
    assert not list(cache.dir.glob("*.tmp"))
    assert cards[0].audio == AudioInfo()  # la entrada no se modifica


def test_shared_text_is_synthesized_once(cache: AudioCache):
    cards = [make_card("run", "VERB", "Run, Forrest."), make_card("run", "NOUN", "Run, Forrest.")]
    tts = FakeTTS()
    out, stats = run(cards, tts, cache)
    assert sorted(tts.texts()) == ["Run, Forrest.", "run"]
    assert out[0].audio == out[1].audio
    assert stats.generated == 2


def test_second_run_uses_cache_and_skips_done_cards(cache: AudioCache):
    cards = [make_card("knife"), make_card("ghost")]
    first, _ = run(cards, FakeTTS(), cache)
    assert not any(needs_work(c, VOICE, cache) for c in first)

    tts = FakeTTS()
    again, stats = run(first, tts, cache)
    assert tts.calls == [] and stats.keys == [] and again == first

    # Mismas tarjetas sin audio en el JSON (p. ej. otro cards.json): se resuelven del caché.
    fresh, stats = run(cards, tts, cache)
    assert tts.calls == [] and stats.from_cache == 4 and stats.generated == 0
    assert fresh == first


def test_regenerates_when_voice_changes_or_file_is_missing(cache: AudioCache):
    first, _ = run([make_card("knife")], FakeTTS(), cache)
    Path(first[0].audio.word).unlink()

    tts = FakeTTS()
    [card], stats = run(first, tts, cache)
    assert tts.texts() == ["knife"] and stats.from_cache == 1  # solo falta la palabra

    tts = FakeTTS()
    [aria], _ = run([card], tts, cache, voice="en-US-AriaNeural")
    assert {v for _, v in tts.calls} == {"en-US-AriaNeural"} and len(tts.calls) == 2
    assert aria.audio.voice == "en-US-AriaNeural" and aria.audio.word != card.audio.word


def test_empty_example_only_gets_word_audio(cache: AudioCache):
    [card], _ = run([make_card("hey", "INTJ", example="<i></i>")], FakeTTS(), cache)
    assert card.audio.word and card.audio.example is None


def test_concurrency_is_bounded(cache: AudioCache):
    cards = [make_card(f"word{i}") for i in range(12)]
    tts = FakeTTS()
    out, stats = run(cards, tts, cache, concurrency=3)
    assert tts.max_active == 3  # en paralelo, pero nunca más de 3
    assert stats.generated == 24 and all(c.audio.word for c in out)


# --- Reintentos ------------------------------------------------------------------------


def test_retries_transient_errors_with_backoff(cache: AudioCache):
    tts = FakeTTS()
    tts.fail["knife"] = [NoAudioReceived("x"), WebSocketError("y"), TimeoutError()]
    sleep = NoSleep()
    [card], _ = run([make_card("knife")], tts, cache, sleep=sleep)
    assert sleep.waits == [2.0, 4.0, 8.0]
    assert tts.texts().count("knife") == 4
    assert Path(card.audio.word).is_file()


def test_gives_up_after_retries_and_cleans_tmp(cache: AudioCache):
    tts = FakeTTS()
    tts.fail["knife"] = [ConnectionError("down")] * 4
    sleep = NoSleep()
    with pytest.raises(AudioError, match="4 intentos"):
        asyncio.run(
            synthesize_with_retry(tts, "knife", VOICE, cache.path_for(VOICE, "knife"), sleep=sleep)
        )
    assert sleep.waits == [2.0, 4.0, 8.0]
    assert not list(cache.dir.iterdir())


def test_empty_file_is_retried(cache: AudioCache):
    tts = FakeTTS()
    tts.empty.add("knife")
    with pytest.raises(AudioError, match="vacío"):
        run([make_card("knife")], tts, cache)
    assert tts.texts().count("knife") == 4


def test_non_transient_error_fails_immediately(cache: AudioCache):
    tts = FakeTTS()
    tts.fail["knife"] = [ValueError("Invalid voice 'nope'.")]
    sleep = NoSleep()
    with pytest.raises(AudioError, match="Invalid voice"):
        run([make_card("knife")], tts, cache, sleep=sleep)
    assert sleep.waits == [] and tts.texts().count("knife") == 1


# --- Checkpoints -----------------------------------------------------------------------


def test_limit_and_periodic_checkpoint(cache: AudioCache):
    cards = [make_card(f"w{i}") for i in range(25)]
    saved: list[int] = []

    def checkpoint(current: list[Card]) -> None:
        saved.append(sum(1 for c in current if c.audio.word))

    out, stats = run(cards, FakeTTS(), cache, checkpoint=checkpoint)
    assert saved == [10, 20] and len(stats.keys) == 25

    saved.clear()
    out, stats = run(cards, FakeTTS(), cache, limit=5, checkpoint=checkpoint)
    assert len(out) == 25 and len(stats.keys) == 5 and saved == []
    assert sum(1 for c in out if c.audio.word) == 5


def test_error_cancels_pending_and_saves_progress(cache: AudioCache):
    cards = [make_card("a"), make_card("b"), make_card("c"), make_card("d")]
    tts = FakeTTS()
    tts.fail["c"] = [ValueError("boom")]
    saved: list[list[Card]] = []
    with pytest.raises(AudioError, match="boom"):
        run(cards, tts, cache, concurrency=1, checkpoint=saved.append)
    [last] = saved
    assert [bool(c.audio.word) for c in last] == [True, True, False, False]
    assert "d" not in tts.texts()  # lo pendiente se canceló
    assert not list(cache.dir.glob("*.tmp"))


# --- edge-tts real (simulado) ----------------------------------------------------------


class FakeCommunicate:
    """Sustituye ``edge_tts.Communicate``: misma firma usada, sin red."""

    created: list[tuple[str, str]] = []

    def __init__(self, text: str, voice: str) -> None:
        if not voice.endswith("Neural"):
            raise ValueError(f"Invalid voice '{voice}'.")
        self.text, self.voice = text, voice
        FakeCommunicate.created.append((text, voice))

    async def save(self, audio_fname: str) -> None:
        Path(audio_fname).write_bytes(b"ID3" + self.text.encode())


@pytest.fixture
def fake_edge(monkeypatch: pytest.MonkeyPatch) -> type[FakeCommunicate]:
    FakeCommunicate.created = []
    monkeypatch.setattr(audio_mod.edge_tts, "Communicate", FakeCommunicate)
    return FakeCommunicate


def test_edge_tts_synthesize_uses_communicate(tmp_path: Path, fake_edge):
    dest = tmp_path / "a.mp3"
    asyncio.run(edge_tts_synthesize("hello", VOICE, dest))
    assert fake_edge.created == [("hello", VOICE)]
    assert dest.read_bytes() == b"ID3hello"


def test_generate_with_default_synthesizer(cache: AudioCache, fake_edge):
    [card], _ = generate_audio([make_card("knife")], cache=cache, voice=VOICE)
    assert sorted(fake_edge.created) == [("I need the knife right now.", VOICE), ("knife", VOICE)]
    assert Path(card.audio.word).read_bytes() == b"ID3knife"


# --- Empaquetado en el .apkg -----------------------------------------------------------


def read_apkg(apkg: Path, tmp_path: Path) -> tuple[dict, dict[str, bytes]]:
    with zipfile.ZipFile(apkg) as zf:
        media = json.loads(zf.read("media"))
        files = {name: zf.read(idx) for idx, name in media.items()}
        db_path = tmp_path / "col.anki2"
        db_path.write_bytes(zf.read("collection.anki2"))
    db = sqlite3.connect(db_path)
    notes = {
        guid: dict(zip(FIELDS, flds.split("\x1f"), strict=True))
        for guid, flds in db.execute("SELECT guid, flds FROM notes")
    }
    db.close()
    return notes, files


def test_build_packs_mp3_and_fills_sound_fields(tmp_path: Path, cache: AudioCache):
    cards = [
        make_card("run", "VERB", "Run, Forrest."),
        make_card("run", "NOUN", "A long run."),
        make_card("ghost"),  # sin audio
    ]
    with_audio, _ = run(cards[:2], FakeTTS(), cache)
    cards[:2] = with_audio

    notes, media = read_apkg(build_deck(cards, tmp_path / "d.apkg"), tmp_path)
    verb, noun, ghost = (notes[note_guid(c)] for c in cards)

    # 3 mp3: "run" compartido + 2 oraciones
    assert len(media) == 3
    assert all(n.startswith("anki-series-") and n.endswith(".mp3") for n in media)
    assert verb["AudioWord"] == noun["AudioWord"]
    assert verb["AudioExample"] != noun["AudioExample"]
    for note in (verb, noun):
        for field in ("AudioWord", "AudioExample"):
            name = note[field].removeprefix("[sound:").removesuffix("]")
            assert note[field] == f"[sound:{name}]" and name in media
    word_name = verb["AudioWord"][len("[sound:") : -1]
    assert media[word_name] == f"mp3:{VOICE}:run".encode()
    assert ghost["AudioWord"] == ghost["AudioExample"] == ""


def test_build_missing_audio_file_fails(tmp_path: Path):
    card = make_card("knife")
    card.audio = AudioInfo(voice=VOICE, word=str(tmp_path / "nope.mp3"))
    with pytest.raises(FileNotFoundError, match="knife/NOUN"):
        build_deck([card], tmp_path / "d.apkg")


# --- CLI -------------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_edge) -> dict:
    data = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.delenv("TTS_VOICE", raising=False)
    cards_json = data / "output" / "cards.json"
    cards_json.parent.mkdir(parents=True)
    cards_json.write_bytes(CARDS.dump_json([make_card("knife"), make_card("ghost")]))
    return {"data": data, "cards_json": cards_json, "fake": fake_edge}


def test_cli_audio_updates_cards_json_in_place(cli_env: dict):
    result = runner.invoke(cli.app, ["audio"])
    assert result.exit_code == 0, result.output
    cards = CARDS.validate_json(cli_env["cards_json"].read_bytes())
    assert all(c.audio.voice == VOICE and Path(c.audio.example).is_file() for c in cards)
    assert "Audios nuevos: 4" in result.output and "Total con audio" in result.output
    assert len(list((cli_env["data"] / "cache" / "audio").glob("*.mp3"))) == 4

    again = runner.invoke(cli.app, ["audio"])
    assert "Procesadas: 0" in again.output and len(cli_env["fake"].created) == 4


def test_cli_audio_voice_limit_and_output(cli_env: dict, tmp_path: Path):
    out = tmp_path / "with-audio.json"
    args = ["audio", "-i", str(cli_env["cards_json"]), "-o", str(out), "--limit", "1"]
    result = runner.invoke(cli.app, [*args, "--voice", "en-US-AriaNeural"])
    assert result.exit_code == 0, result.output
    cards = CARDS.validate_json(out.read_bytes())
    assert cards[0].audio.voice == "en-US-AriaNeural" and cards[1].audio.word is None
    assert "1 sin audio" in result.output
    assert {v for _, v in cli_env["fake"].created} == {"en-US-AriaNeural"}
    # la entrada no se tocó
    assert CARDS.validate_json(cli_env["cards_json"].read_bytes())[0].audio == AudioInfo()


def test_cli_audio_voice_from_env(cli_env: dict, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TTS_VOICE", "en-GB-SoniaNeural")
    assert runner.invoke(cli.app, ["audio"]).exit_code == 0
    assert {v for _, v in cli_env["fake"].created} == {"en-GB-SoniaNeural"}


def test_cli_audio_invalid_voice_fails_cleanly(cli_env: dict):
    result = runner.invoke(cli.app, ["audio", "--voice", "nope"])
    assert result.exit_code == 1
    assert "Invalid voice" in result.output and "vuelve a correr" in result.output


def test_cli_audio_missing_input(tmp_path: Path):
    result = runner.invoke(cli.app, ["audio", "-i", str(tmp_path / "nope.json")])
    assert result.exit_code == 1 and "translate" in result.output


def test_cli_audio_then_build_packs_mp3(cli_env: dict):
    assert runner.invoke(cli.app, ["audio"]).exit_code == 0
    result = runner.invoke(cli.app, ["build"])
    assert result.exit_code == 0, result.output
    assert "Con audio: 2" in result.output
    with zipfile.ZipFile(cli_env["data"] / "output" / "x.apkg") as zf:
        names = json.loads(zf.read("media")).values()
    assert len(names) == 4 and all(n.endswith(".mp3") for n in names)
