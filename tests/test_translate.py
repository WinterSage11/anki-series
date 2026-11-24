"""Tests de la Fase 4. Cero llamadas reales: el cliente de OpenAI es un doble en memoria."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from anki_series import cli
from anki_series.models import Card, SourceRef
from anki_series.translate import (
    TranslationBatch,
    TranslationCache,
    TranslationError,
    WordTranslation,
    request_batch,
    translate_cards,
)

MODEL = "gpt-test"
CARDS = TypeAdapter(list[Card])


def make_card(lemma: str, pos: str = "NOUN", rank: int = 1, sentence: str | None = None) -> Card:
    return Card(
        series="X",
        lemma=lemma,
        pos=pos,
        rank=rank,
        frequency=3,
        example_en=sentence or f"I need the {lemma} right now.",
        source=SourceRef(file="e01.srt", start="00:00:01,000"),
    )


def completion(parsed: TranslationBatch | None, refusal: str | None = None) -> SimpleNamespace:
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Imita ``client.chat.completions.parse``. Por defecto traduce todo lo que recibe.

    ``script`` es una lista de respuestas/excepciones que se consumen en orden; ``None``
    en el script (o script agotado) = comportamiento por defecto.
    """

    def __init__(self, script: list | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0) if self.script else None
        if isinstance(step, Exception):
            raise step
        if step is not None:
            return step
        items = json.loads(kwargs["messages"][1]["content"])["items"]
        return completion(
            TranslationBatch(
                items=[
                    WordTranslation(
                        id=it["id"],
                        translation=f"es-{it['word']}",
                        example_es=f"Oración con {it['word']}.",
                        imageable=it["word"] == "car",
                        image_query="red car" if it["word"] == "car" else None,
                    )
                    for it in items
                ]
            )
        )


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx2.Request("POST", "https://api.test"))


def status_error(cls: type[openai.APIStatusError], code: int) -> openai.APIStatusError:
    request = httpx2.Request("POST", "https://api.test")
    return cls("boom", response=httpx2.Response(code, request=request), body=None)


def no_sleep(_: float) -> None:
    pass


# --- request_batch: parseo estructurado ---


def test_request_batch_sends_schema_and_items():
    client = FakeClient()
    cards = [make_card("car"), make_card("bail", "VERB")]
    results = request_batch(client, MODEL, cards)

    call = client.calls[0]
    assert call["model"] == MODEL
    assert call["response_format"] is TranslationBatch
    sent = json.loads(call["messages"][1]["content"])["items"]
    assert sent[1] == {"id": 1, "word": "bail", "pos": "VERB", "sentence": cards[1].example_en}
    assert results[0].translation == "es-car"
    assert results[0].imageable is True and results[0].image_query == "red car"
    assert results[1].imageable is False and results[1].image_query is None


def test_request_batch_normalizes_inconsistent_image_fields():
    parsed = TranslationBatch(
        items=[
            WordTranslation(
                id=0, translation=" coche ", example_es="x", imageable=True, image_query="  "
            ),
            WordTranslation(
                id=1, translation="huir", example_es="y", imageable=False, image_query="run away"
            ),
        ]
    )
    client = FakeClient([completion(parsed)])
    results = request_batch(client, MODEL, [make_card("car"), make_card("flee", "VERB")])
    assert results[0].translation == "coche"
    assert (results[0].imageable, results[0].image_query) == (False, None)
    assert (results[1].imageable, results[1].image_query) == (False, None)


def test_request_batch_ignores_unknown_ids():
    parsed = TranslationBatch(
        items=[
            WordTranslation(
                id=7, translation="a", example_es="b", imageable=False, image_query=None
            )
        ]
    )
    results = request_batch(FakeClient([completion(parsed)]), MODEL, [make_card("car")])
    assert results == {}


def test_request_batch_refusal_raises():
    client = FakeClient([completion(None, refusal="no")])
    with pytest.raises(TranslationError, match="negó"):
        request_batch(client, MODEL, [make_card("car")])


def test_request_batch_empty_parse_raises():
    with pytest.raises(TranslationError, match="vacía"):
        request_batch(FakeClient([completion(None)]), MODEL, [make_card("car")])


# --- request_batch: reintentos ---


def test_retries_transient_errors_with_backoff():
    waits: list[float] = []
    client = FakeClient([connection_error(), status_error(openai.RateLimitError, 429)])
    results = request_batch(client, MODEL, [make_card("car")], sleep=waits.append)
    assert results[0].translation == "es-car"
    assert len(client.calls) == 3
    assert waits == [2.0, 4.0]


def test_gives_up_after_max_retries():
    errors = [status_error(openai.InternalServerError, 500) for _ in range(3)]
    client = FakeClient(errors)
    with pytest.raises(TranslationError, match="3 intentos"):
        request_batch(client, MODEL, [make_card("car")], max_retries=2, sleep=no_sleep)
    assert len(client.calls) == 3


def test_non_transient_error_fails_fast():
    client = FakeClient([status_error(openai.AuthenticationError, 401)])
    with pytest.raises(TranslationError, match="OpenAI"):
        request_batch(client, MODEL, [make_card("car")], sleep=no_sleep)
    assert len(client.calls) == 1


# --- caché y orquestación ---


def test_translate_cards_fills_fields_in_batches(tmp_path: Path):
    cache = TranslationCache(tmp_path / "t.json")
    client = FakeClient()
    cards = [make_card(w, rank=i + 1) for i, w in enumerate(["car", "key", "gun", "cop", "job"])]
    out, new = translate_cards(
        cards, cache=cache, model=MODEL, client_factory=lambda: client, batch_size=2
    )
    assert new == 5
    assert len(client.calls) == 3  # 2 + 2 + 1
    assert [c.lemma for c in out] == ["car", "key", "gun", "cop", "job"]
    car = out[0]
    assert (car.translation, car.example_es) == ("es-car", "Oración con car.")
    assert car.imageable is True and car.image.query == "red car"
    assert out[1].imageable is False and out[1].image.query is None


def test_cache_avoids_second_call_and_needs_no_client(tmp_path: Path):
    path = tmp_path / "cache" / "t.json"
    cards = [make_card("car"), make_card("key")]
    translate_cards(cards, cache=TranslationCache(path), model=MODEL, client_factory=FakeClient)
    assert path.is_file()

    def boom():
        raise AssertionError("no debería crear cliente")

    out, new = translate_cards(
        cards, cache=TranslationCache(path), model=MODEL, client_factory=boom
    )
    assert new == 0
    assert all(c.translation for c in out)


def test_cache_key_depends_on_sentence_and_model():
    a = make_card("car", sentence="Get in the car.")
    b = make_card("car", sentence="The car is red.")
    assert TranslationCache.key(a, MODEL) != TranslationCache.key(b, MODEL)
    assert TranslationCache.key(a, MODEL) != TranslationCache.key(a, "other-model")
    # La serie no forma parte de la clave.
    assert TranslationCache.key(a, MODEL) == TranslationCache.key(
        a.model_copy(update={"series": "Y"}), MODEL
    )


def test_failure_midway_keeps_previous_batches_in_cache(tmp_path: Path):
    path = tmp_path / "t.json"
    cards = [make_card(w) for w in ["car", "key", "gun"]]
    auth = status_error(openai.AuthenticationError, 401)
    first = FakeClient([None, auth])  # lote 1 OK, lote 2 falla

    with pytest.raises(TranslationError):
        translate_cards(
            cards,
            cache=TranslationCache(path),
            model=MODEL,
            client_factory=lambda: first,
            batch_size=1,
        )
    assert len(TranslationCache(path)) == 1

    resume = FakeClient()
    out, new = translate_cards(
        cards,
        cache=TranslationCache(path),
        model=MODEL,
        client_factory=lambda: resume,
        batch_size=1,
    )
    assert new == 2 and len(resume.calls) == 2
    assert all(c.translation for c in out)


def test_omitted_card_stays_untranslated_and_uncached(tmp_path: Path):
    partial = TranslationBatch(
        items=[
            WordTranslation(
                id=0, translation="coche", example_es="x", imageable=False, image_query=None
            )
        ]
    )
    cache = TranslationCache(tmp_path / "t.json")
    out, new = translate_cards(
        [make_card("car"), make_card("key")],
        cache=cache,
        model=MODEL,
        client_factory=lambda: FakeClient([completion(partial)]),
    )
    assert new == 1
    assert out[0].translation == "coche" and out[1].translation is None
    assert len(cache) == 1


# --- CLI ---

runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """DATA_DIR temporal, API key falsa y cliente de OpenAI reemplazado por el doble."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", MODEL)
    state: dict = {"clients": []}

    def fake_openai(**kwargs):
        client = FakeClient()
        state["clients"].append((kwargs, client))
        return client

    monkeypatch.setattr(cli, "OpenAI", fake_openai)
    words = tmp_path / "words.json"
    cards = [make_card(w, rank=i + 1) for i, w in enumerate(["car", "key", "gun"])]
    words.write_bytes(CARDS.dump_json(cards))
    state["words"] = words
    state["tmp"] = tmp_path
    return state


def test_cli_translate_writes_cards_json(cli_env: dict):
    out = cli_env["tmp"] / "cards.json"
    args = ["translate", "-i", str(cli_env["words"]), "-o", str(out), "--batch-size", "2"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    cards = CARDS.validate_json(out.read_bytes())
    assert [c.translation for c in cards] == ["es-car", "es-key", "es-gun"]
    assert "Traducidas: 3/3 (nuevas: 3, de caché: 0)" in result.output
    kwargs, client = cli_env["clients"][0]
    assert kwargs == {"api_key": "sk-test", "max_retries": 0}
    assert len(client.calls) == 2 and client.calls[0]["model"] == MODEL
    assert (cli_env["tmp"] / "data" / "cache" / "translations.json").is_file()

    again = runner.invoke(cli.app, args)
    assert "nuevas: 0, de caché: 3" in again.output
    assert len(cli_env["clients"]) == 1  # segunda corrida sin crear cliente


def test_cli_translate_limit_and_default_output(cli_env: dict):
    result = runner.invoke(
        cli.app, ["translate", "-i", str(cli_env["words"]), "--limit", "2", "--model", "m2"]
    )
    assert result.exit_code == 0, result.output
    out = cli_env["tmp"] / "data" / "output" / "cards.json"
    assert len(CARDS.validate_json(out.read_bytes())) == 2
    assert cli_env["clients"][0][1].calls[0]["model"] == "m2"


def test_cli_translate_missing_key(cli_env: dict, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    result = runner.invoke(cli.app, ["translate", "-i", str(cli_env["words"])])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output
    assert cli_env["clients"] == []


def test_cli_translate_missing_input(cli_env: dict):
    result = runner.invoke(cli.app, ["translate", "-i", str(cli_env["tmp"] / "nope.json")])
    assert result.exit_code == 1
    assert "extract" in result.output


def test_cli_translate_api_error_fails_cleanly(cli_env: dict, monkeypatch: pytest.MonkeyPatch):
    def failing_openai(**kwargs):
        return FakeClient([status_error(openai.AuthenticationError, 401)])

    monkeypatch.setattr(cli, "OpenAI", failing_openai)
    out = cli_env["tmp"] / "cards.json"
    result = runner.invoke(cli.app, ["translate", "-i", str(cli_env["words"]), "-o", str(out)])
    assert result.exit_code == 1
    assert "caché" in result.output
    assert not out.exists()
