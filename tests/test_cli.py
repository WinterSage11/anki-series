import json
import zipfile
from pathlib import Path

import httpx
import pytest
from apkg_factory import make_apkg
from pydantic import TypeAdapter

# Dobles de red de cada fase (sin llamadas reales): se reutilizan tal cual.
from test_audio import FakeCommunicate
from test_images import FakePixabay
from test_translate import FakeClient
from typer.testing import CliRunner

from anki_series import __version__, audio, cli
from anki_series.cli import app
from anki_series.models import Card, ImageStatus

FIXTURE = Path(__file__).parent / "fixtures" / "sample.srt"
EXPR_FIXTURE = Path(__file__).parent / "fixtures" / "expressions.srt"
CARDS = TypeAdapter(list[Card])
runner = CliRunner()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_extract_dry_run_shows_summary_and_samples():
    result = runner.invoke(
        app, ["extract", str(FIXTURE), "--series", "The Night Shift", "--dry-run", "--limit", "2"]
    )
    assert result.exit_code == 0, result.output
    assert "Archivos: 1 · Oraciones: 11" in result.output
    assert "[sample.srt 00:00:07,200] We need to bail before the cops show up." in result.output
    assert "Relax, man." not in result.output  # --limit 2


def test_extract_requires_series():
    result = runner.invoke(app, ["extract", str(FIXTURE), "--dry-run"])
    assert result.exit_code != 0


def test_extract_writes_words_json(tmp_path: Path):
    out = tmp_path / "out" / "words.json"
    result = runner.invoke(
        app, ["extract", str(FIXTURE), "--series", "The Night Shift", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    cards = TypeAdapter(list[Card]).validate_json(out.read_bytes())
    assert cards and cards[0].rank == 1
    assert all(c.series == "The Night Shift" for c in cards)
    assert ("bail", "VERB") in {(c.lemma, c.pos) for c in cards}
    assert f"Lemas: {len(cards)}" in result.output


def test_extract_excludes_known_deck(tmp_path: Path):
    apkg = make_apkg(tmp_path, notes=[["1", "bail", ""], ["2", "keys", ""]])
    out = tmp_path / "words.json"
    args = ["extract", str(FIXTURE), "--series", "X", "-o", str(out), "--known-deck", str(apkg)]
    result = runner.invoke(app, [*args, "--known-field", "Palabra"])
    assert result.exit_code == 0, result.output
    lemmas = {c.lemma for c in TypeAdapter(list[Card]).validate_json(out.read_bytes())}
    assert "bail" not in lemmas and "key" not in lemmas
    assert "car" in lemmas


def test_extract_known_deck_wrong_field_fails_cleanly(tmp_path: Path):
    apkg = make_apkg(tmp_path)
    args = ["extract", str(FIXTURE), "--series", "X", "-o", str(tmp_path / "w.json")]
    result = runner.invoke(app, [*args, "--known-deck", str(apkg), "--known-field", "0"])
    assert result.exit_code == 1
    assert "Campos disponibles" in result.output
    assert not (tmp_path / "w.json").exists()


def test_extract_exclude_common(tmp_path: Path):
    out = tmp_path / "words.json"
    args = ["extract", str(FIXTURE), "--series", "X", "-o", str(out)]
    result = runner.invoke(app, [*args, "--exclude-common", "200"])
    assert result.exit_code == 0, result.output
    assert "top 200" in result.output
    lemmas = {c.lemma for c in CARDS.validate_json(out.read_bytes())}
    assert not {"need", "find", "time", "go"} & lemmas  # entre las 200 más comunes
    assert {"bail", "cop", "errand"} <= lemmas


def test_extract_no_interjections_or_contraction_nouns(tmp_path: Path):
    out = tmp_path / "words.json"
    result = runner.invoke(app, ["extract", str(FIXTURE), "--series", "X", "-o", str(out)])
    assert result.exit_code == 0, result.output
    found = {(c.lemma, c.pos) for c in CARDS.validate_json(out.read_bytes())}
    assert not any(pos == "INTJ" for _lemma, pos in found)  # "huh"
    assert ("go", "NOUN") not in found
    assert ("go", "VERB") in found  # "gonna"


# --- all: flujo completo con OpenAI, Pixabay y edge-tts simulados ----------------------


@pytest.fixture
def all_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    FakeCommunicate.created = []
    monkeypatch.setattr(audio.edge_tts, "Communicate", FakeCommunicate)
    data = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("PIXABAY_API_KEY", "test-key")
    monkeypatch.delenv("TTS_VOICE", raising=False)
    openai_clients: list[FakeClient] = []

    def fake_openai(**kwargs):
        openai_clients.append(FakeClient())
        return openai_clients[-1]

    pixabay = FakePixabay({"red car": ["car, red, vehicle"]})
    monkeypatch.setattr(cli, "OpenAI", fake_openai)
    monkeypatch.setattr(
        cli, "_pixabay_http", lambda: httpx.Client(transport=httpx.MockTransport(pixabay.handler))
    )
    return {"data": data, "openai": openai_clients, "pixabay": pixabay, "tts": FakeCommunicate}


def test_all_runs_full_pipeline(all_env: dict):
    args = ["all", str(FIXTURE), "--series", "The Night Shift", "--exclude-common", "100"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    for step in ("[1/5] extract", "[2/5] translate", "[3/5] images", "[4/5] audio", "[5/5] build"):
        assert step in result.output

    out = all_env["data"] / "output"
    cards = CARDS.validate_json((out / "cards.json").read_bytes())
    words = CARDS.validate_json((out / "words.json").read_bytes())
    assert len(cards) == len(words) > 0
    assert all(c.translation and c.audio.word and c.audio.example for c in cards)
    car = next(c for c in cards if c.lemma == "car")
    assert car.image.status == ImageStatus.FOUND
    assert all_env["pixabay"].searches == ["red car"]

    apkg = out / "the-night-shift.apkg"
    with zipfile.ZipFile(apkg) as zf:
        media = json.loads(zf.read("media"))
    assert len(media) > 1  # imagen + audios empaquetados


def test_all_second_run_uses_caches(all_env: dict):
    args = ["all", str(FIXTURE), "--series", "X"]
    assert runner.invoke(cli.app, args).exit_code == 0
    tts_calls = len(all_env["tts"].created)
    searches = list(all_env["pixabay"].searches)

    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert len(all_env["openai"]) == 1  # 2.ª corrida: todo de caché, ni se crea el cliente
    assert all_env["pixabay"].searches == searches
    assert len(all_env["tts"].created) == tts_calls


def test_all_limit_and_options(all_env: dict):
    args = ["all", str(FIXTURE), "--series", "X", "--limit", "3", "--model", "gpt-other"]
    result = runner.invoke(cli.app, [*args, "--voice", "en-US-AriaNeural", "--concurrency", "2"])
    assert result.exit_code == 0, result.output
    cards = CARDS.validate_json((all_env["data"] / "output" / "cards.json").read_bytes())
    assert len(cards) == 3
    assert all(c.audio.voice == "en-US-AriaNeural" for c in cards)
    assert all_env["openai"][0].calls[0]["model"] == "gpt-other"
    assert "Notas: 3" in result.output


def test_all_stops_on_failing_stage(all_env: dict, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    result = runner.invoke(cli.app, ["all", str(FIXTURE), "--series", "X"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output
    assert "[3/5] images" not in result.output
    assert (all_env["data"] / "output" / "words.json").exists()


# --- extract --expressions -------------------------------------------------------------


def test_extract_without_expressions_stays_offline(all_env: dict):
    result = runner.invoke(app, ["extract", str(EXPR_FIXTURE), "--series", "X"])
    assert result.exit_code == 0, result.output
    assert all_env["openai"] == []  # ni se crea el cliente
    cards = CARDS.validate_json((all_env["data"] / "output" / "words.json").read_bytes())
    assert not {c.pos for c in cards} & {"phrasal", "idiom"}
    assert "Expresiones" not in result.output


def test_extract_expressions_quota(all_env: dict):
    args = ["extract", str(EXPR_FIXTURE), "--series", "X", "--top", "10", "--expressions", "3"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    cards = CARDS.validate_json((all_env["data"] / "output" / "words.json").read_bytes())
    exprs = [c for c in cards if c.pos in {"phrasal", "idiom"}]
    assert len(cards) == 10 and len(exprs) == 3  # 3 expresiones + 7 palabras
    assert "sit down" not in {c.lemma for c in exprs}  # literal: descartada por el filtro
    assert [c.rank for c in cards] == list(range(1, 11))
    assert "Lemas: 7 · Expresiones: 3" in result.output
    assert (all_env["data"] / "cache" / "expressions.json").exists()
    assert all_env["openai"][0].calls[0]["model"] == "gpt-test"


def test_extract_expressions_fewer_than_quota_fills_with_words(all_env: dict):
    args = ["extract", str(EXPR_FIXTURE), "--series", "X", "--top", "20", "--expressions", "50"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    cards = CARDS.validate_json((all_env["data"] / "output" / "words.json").read_bytes())
    exprs = [c for c in cards if c.pos in {"phrasal", "idiom"}]
    assert len(exprs) == 4 and len(cards) == 20


def test_extract_expressions_excludes_known_deck_but_not_wordfreq(all_env: dict, tmp_path: Path):
    apkg = make_apkg(tmp_path, notes=[["1", "get rid of", ""], ["2", "car", ""]])
    args = ["extract", str(EXPR_FIXTURE), "--series", "X", "--expressions", "10"]
    args += ["--known-deck", str(apkg), "--known-field", "Palabra", "--exclude-common", "500"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    cards = CARDS.validate_json((all_env["data"] / "output" / "words.json").read_bytes())
    lemmas = {c.lemma for c in cards}
    assert "get rid of" not in lemmas and "car" not in lemmas
    # "make sense" son palabras muy comunes, pero las expresiones no pasan por wordfreq.
    assert {"figure out", "make sense", "no offense"} <= lemmas


def test_extract_expressions_missing_key_fails_cleanly(all_env: dict, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    args = ["extract", str(EXPR_FIXTURE), "--series", "X", "--expressions", "5"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_all_with_expressions(all_env: dict):
    args = ["all", str(EXPR_FIXTURE), "--series", "X", "--top", "12", "--expressions", "4"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    cards = CARDS.validate_json((all_env["data"] / "output" / "cards.json").read_bytes())
    exprs = {c.lemma: c for c in cards if c.pos in {"phrasal", "idiom"}}
    assert set(exprs) == {"figure out", "get rid of", "make sense", "no offense"}
    assert exprs["figure out"].translation == "es-figure out"
    # Un cliente compartido: 1 llamada al filtro + lotes de traducción (palabras y expresiones).
    prompts = [c["messages"][0]["content"] for cl in all_env["openai"] for c in cl.calls]
    assert any("curate" in p for p in prompts) and any("English expressions" in p for p in prompts)


def test_extract_missing_path_fails_cleanly(tmp_path: Path):
    result = runner.invoke(
        app, ["extract", str(tmp_path / "nope.srt"), "--series", "X", "--dry-run"]
    )
    assert result.exit_code == 1
    assert "Error" in result.output
