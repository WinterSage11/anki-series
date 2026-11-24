"""Tests de la Fase 5. Cero llamadas reales: Pixabay se simula con ``httpx.MockTransport``."""

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest
from PIL import Image
from pydantic import TypeAdapter
from typer.testing import CliRunner

from anki_series import cli
from anki_series.images import (
    ImageCache,
    ImageError,
    OverridesError,
    PixabayClient,
    fetch_images,
    is_relevant,
    load_overrides,
    save_jpeg,
)
from anki_series.models import Card, ImageInfo, ImageStatus, SourceRef

CARDS = TypeAdapter(list[Card])
runner = CliRunner()


def png_bytes(size: tuple[int, int] = (1200, 800), mode: str = "RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, "red").save(buffer, "PNG")
    return buffer.getvalue()


def make_card(
    lemma: str,
    pos: str = "NOUN",
    *,
    imageable: bool | None = True,
    query: str | None = None,
    status: ImageStatus = ImageStatus.PENDING,
) -> Card:
    if imageable and query is None:
        query = lemma
    return Card(
        series="X",
        lemma=lemma,
        pos=pos,
        rank=1,
        frequency=3,
        example_en=f"I need the {lemma} right now.",
        source=SourceRef(file="e01.srt", start="00:00:01,000"),
        translation="algo",
        example_es="Necesito eso ya.",
        imageable=imageable,
        image=ImageInfo(status=status, query=query),
    )


class FakePixabay:
    """Simula la API (búsqueda + CDN). ``results`` mapea query → lista de tags de cada hit."""

    def __init__(self, results: dict[str, list[str]] | None = None) -> None:
        self.results = results or {}
        self.searches: list[str] = []
        self.downloads: list[str] = []
        self.fail_next: list[int] = []  # códigos a devolver antes de responder bien
        self.missing: set[str] = set()  # URLs de imagen que devuelven 404

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0))
        if request.url.host == "pixabay.com":
            if request.url.params["key"] != "test-key":
                return httpx.Response(400, text="[ERROR 400] Invalid or missing API key")
            query = request.url.params["q"]
            self.searches.append(query)
            hits = [
                {
                    "webformatURL": f"https://cdn.pixabay.test/{query.replace(' ', '_')}_{i}.png",
                    "pageURL": f"https://pixabay.com/photos/{i}/",
                    "user": f"user{i}",
                    "tags": tags,
                }
                for i, tags in enumerate(self.results.get(query, []))
            ]
            return httpx.Response(200, json={"total": len(hits), "hits": hits})
        url = str(request.url)
        self.downloads.append(url)
        if url in self.missing:
            return httpx.Response(404)
        return httpx.Response(200, content=png_bytes())

    def client(self, key: str = "test-key") -> PixabayClient:
        http = httpx.Client(transport=httpx.MockTransport(self.handler))
        return PixabayClient(http, key, sleep=lambda _: None)


@pytest.fixture
def cache(tmp_path: Path) -> ImageCache:
    return ImageCache(tmp_path / "cache" / "images")


def run(cards, fake: FakePixabay, cache: ImageCache, overrides=None, **kw):
    return fetch_images(
        cards, cache=cache, overrides=overrides or {}, client_factory=fake.client, **kw
    )


# --- Relevancia y búsqueda ---


@pytest.mark.parametrize(
    ("query", "tags", "expected"),
    [
        ("knife", "knife, kitchen, steel", True),
        ("car", "cars, road", True),
        ("police officer", "policeman, uniform", True),
        ("bat animal flying", "bat, wildlife", True),
        ("knife", "sunset, beach, sea", False),
    ],
)
def test_is_relevant(query: str, tags: str, expected: bool):
    assert is_relevant(query, tags) is expected


def test_search_filters_irrelevant_hits():
    fake = FakePixabay({"knife": ["sunset, beach", "knife, kitchen"]})
    hits = fake.client().search("knife")
    assert [h.tags for h in hits] == ["knife, kitchen"]


def test_search_invalid_key_fails_without_leaking_it():
    fake = FakePixabay()
    with pytest.raises(ImageError, match="Invalid or missing API key") as info:
        fake.client(key="secret-123").search("knife")
    assert "secret-123" not in str(info.value)


def test_search_retries_transient_errors():
    fake = FakePixabay({"knife": ["knife"]})
    fake.fail_next = [429, 503]
    assert len(fake.client().search("knife")) == 1


def test_search_gives_up_after_retries():
    fake = FakePixabay()
    fake.fail_next = [500] * 10
    with pytest.raises(ImageError, match="4 intentos"):
        fake.client().search("knife")


# --- Redimensionado ---


def test_save_jpeg_resizes_and_converts(tmp_path: Path):
    out = save_jpeg(png_bytes((1200, 800), "RGBA"), tmp_path / "a.jpg")
    with Image.open(out) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"
        assert img.size == (400, 267)


def test_save_jpeg_keeps_small_images(tmp_path: Path):
    with Image.open(save_jpeg(png_bytes((100, 50)), tmp_path / "a.jpg")) as img:
        assert img.size == (100, 50)


def test_save_jpeg_rejects_garbage(tmp_path: Path):
    with pytest.raises(ImageError):
        save_jpeg(b"not an image", tmp_path / "a.jpg")


# --- Orquestación y caché ---


def test_fetch_found_records_metadata(cache: ImageCache):
    fake = FakePixabay({"knife": ["knife, kitchen"]})
    [card], stats = run([make_card("knife")], fake, cache)
    assert card.image.status == ImageStatus.FOUND
    assert Path(card.image.path).is_file()
    assert card.image.source == "pixabay"
    assert card.image.source_url == "https://pixabay.com/photos/0/"
    assert card.image.credit == "user0 / Pixabay"
    assert stats.found == 1


def test_fetch_no_results_marks_not_found(cache: ImageCache):
    fake = FakePixabay({"knife": ["sunset, beach"]})  # solo resultados irrelevantes
    [card], stats = run([make_card("knife")], fake, cache)
    assert card.image.status == ImageStatus.NOT_FOUND
    assert card.image.path is None
    assert fake.downloads == []
    assert stats.not_found == 1


def test_fetch_skips_broken_download_and_uses_next_hit(cache: ImageCache):
    fake = FakePixabay({"knife": ["knife", "knife, kitchen"]})
    fake.missing = {"https://cdn.pixabay.test/knife_0.png"}
    [card], _ = run([make_card("knife")], fake, cache)
    assert card.image.status == ImageStatus.FOUND
    assert card.image.credit == "user1 / Pixabay"


def test_fetch_uses_cache_on_second_run(tmp_path: Path):
    fake = FakePixabay({"knife": ["knife"], "ghost": []})
    cards = [make_card("knife"), make_card("ghost")]
    run(cards, fake, ImageCache(tmp_path / "img"))
    assert fake.searches == ["knife", "ghost"]

    # Otra corrida (caché recargado de disco) sobre las tarjetas originales: cero llamadas.
    def no_client() -> PixabayClient:
        raise AssertionError("no debería llamar a la API")

    out, stats = fetch_images(
        cards, cache=ImageCache(tmp_path / "img"), overrides={}, client_factory=no_client
    )
    assert [c.image.status for c in out] == [ImageStatus.FOUND, ImageStatus.NOT_FOUND]
    assert stats.from_cache == 2


def test_fetch_same_query_searched_once(cache: ImageCache):
    fake = FakePixabay({"run": ["run, jogging"]})
    cards = [make_card("run", "VERB", query="run"), make_card("run", "NOUN", query="run")]
    out, _ = run(cards, fake, cache)
    assert fake.searches == ["run"]
    assert out[0].image.path == out[1].image.path


def test_fetch_selects_only_pending_and_translated(cache: ImageCache):
    fake = FakePixabay({"knife": ["knife"]})
    done = make_card("car", status=ImageStatus.FOUND)
    cards = [
        make_card("knife"),
        make_card("damn", "INTJ", imageable=False),
        make_card("idea", imageable=None),  # sin traducir: no se toca
        done,
    ]
    out, stats = run(cards, fake, cache)
    assert [c.image.status for c in out] == [
        ImageStatus.FOUND,
        ImageStatus.SKIPPED,
        ImageStatus.PENDING,
        ImageStatus.FOUND,
    ]
    assert out[1].image.reason == "no ilustrable"
    assert out[3] == done
    assert fake.searches == ["knife"]
    assert stats.skipped == 1


def test_fetch_without_work_never_creates_client(cache: ImageCache):
    def no_client() -> PixabayClient:
        raise AssertionError("no debería crear el cliente")

    cards = [make_card("damn", imageable=False)]
    out, _ = fetch_images(cards, cache=cache, overrides={}, client_factory=no_client)
    assert out[0].image.status == ImageStatus.SKIPPED


def test_fetch_limit_and_checkpoint(cache: ImageCache):
    fake = FakePixabay({q: [q] for q in ("a1", "b2", "c3")})
    cards = [make_card("a1"), make_card("b2"), make_card("c3")]
    saved: list[list[Card]] = []
    out, stats = run(cards, fake, cache, limit=2, checkpoint=saved.append, checkpoint_every=1)
    assert [c.image.status for c in out] == [
        ImageStatus.FOUND,
        ImageStatus.FOUND,
        ImageStatus.PENDING,
    ]
    assert len(saved) == 2 and len(saved[-1]) == 3  # el checkpoint guarda todas las tarjetas


def test_fetch_error_saves_progress(cache: ImageCache):
    fake = FakePixabay({"a1": ["a1"]})
    cards = [make_card("a1"), make_card("b2")]
    saved: list[list[Card]] = []
    original_handler = fake.handler

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("q") == "b2":
            return httpx.Response(500)
        return original_handler(request)

    fake.handler = handler
    with pytest.raises(ImageError):
        run(cards, fake, cache, checkpoint=saved.append, checkpoint_every=100)
    assert saved[-1][0].image.status == ImageStatus.FOUND
    assert saved[-1][1].image.status == ImageStatus.PENDING


# --- Overrides ---


def write_overrides(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "overrides.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_overrides_missing_file_is_empty(tmp_path: Path):
    assert load_overrides(tmp_path / "nope.yaml") == {}


def test_load_overrides_normalizes_keys(tmp_path: Path):
    path = write_overrides(tmp_path, "Bat/noun: {query: bat animal}\nKnife: {skip: true}\n")
    assert set(load_overrides(path)) == {"bat/NOUN", "knife"}


@pytest.mark.parametrize(
    "text",
    [
        "knife: {}\n",
        "knife: {query: x, skip: true}\n",
        "knife: {color: red}\n",
        "- knife\n",
        "knife: [\n",
    ],
)
def test_load_overrides_invalid(tmp_path: Path, text: str):
    with pytest.raises(OverridesError):
        load_overrides(write_overrides(tmp_path, text))


def test_override_query_takes_precedence_by_pos(tmp_path: Path, cache: ImageCache):
    path = write_overrides(
        tmp_path, "bat: {query: baseball bat}\nbat/NOUN: {query: bat animal flying}\n"
    )
    fake = FakePixabay({"bat animal flying": ["bat, animal"], "baseball bat": ["baseball, bat"]})
    out, _ = run(
        [make_card("bat", "NOUN", query="bat"), make_card("bat", "VERB", imageable=False)],
        fake,
        cache,
        load_overrides(path),
    )
    assert fake.searches == ["bat animal flying", "baseball bat"]
    assert out[0].image.query == "bat animal flying"
    assert out[1].image.status == ImageStatus.FOUND  # el override aplica aunque no sea ilustrable


def test_override_skip_and_local_image(tmp_path: Path, cache: ImageCache):
    local = tmp_path / "mine.png"
    local.write_bytes(png_bytes((900, 900)))
    path = write_overrides(tmp_path, f"knife: {{skip: true}}\ncar: {{image: '{local}'}}\n")
    done = make_card("knife", status=ImageStatus.FOUND)  # override aplica aunque ya tenga imagen
    fake = FakePixabay()
    out, _ = run([done, make_card("car")], fake, cache, load_overrides(path))
    assert out[0].image.status == ImageStatus.SKIPPED
    assert out[0].image.path is None
    assert out[1].image.status == ImageStatus.FOUND
    assert out[1].image.source == "local"
    with Image.open(out[1].image.path) as img:
        assert img.size == (400, 400)
    assert fake.searches == []


def test_override_local_image_missing(tmp_path: Path, cache: ImageCache):
    path = write_overrides(tmp_path, "car: {image: nope.jpg}\n")
    with pytest.raises(OverridesError, match="no encontrada"):
        run([make_card("car")], FakePixabay(), cache, load_overrides(path))


# --- CLI ---


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    data = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("PIXABAY_API_KEY", "test-key")
    fake = FakePixabay({"knife": ["knife, kitchen"], "ghost": []})
    monkeypatch.setattr(
        cli, "_pixabay_http", lambda: httpx.Client(transport=httpx.MockTransport(fake.handler))
    )
    cards_json = data / "output" / "cards.json"
    cards_json.parent.mkdir(parents=True)
    cards = [
        make_card("knife"),
        make_card("ghost"),
        make_card("damn", "INTJ", imageable=False),
    ]
    cards_json.write_bytes(CARDS.dump_json(cards))
    return {"data": data, "cards_json": cards_json, "fake": fake}


def test_cli_images_updates_cards_json_in_place(cli_env: dict):
    result = runner.invoke(cli.app, ["images"])
    assert result.exit_code == 0, result.output
    cards = CARDS.validate_json(cli_env["cards_json"].read_bytes())
    assert [c.image.status for c in cards] == [
        ImageStatus.FOUND,
        ImageStatus.NOT_FOUND,
        ImageStatus.SKIPPED,
    ]
    assert "Con imagen: 1" in result.output
    assert list((cli_env["data"] / "cache" / "images").glob("*.jpg"))


def test_cli_images_limit_and_output(cli_env: dict, tmp_path: Path):
    out = tmp_path / "with-images.json"
    args = ["images", "-i", str(cli_env["cards_json"]), "-o", str(out), "--limit", "1"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    cards = CARDS.validate_json(out.read_bytes())
    assert len(cards) == 3
    assert cards[0].image.status == ImageStatus.FOUND
    assert cards[1].image.status == ImageStatus.PENDING
    assert "1 ilustrables pendientes" in result.output
    assert cli_env["fake"].searches == ["knife"]


def test_cli_images_missing_key(cli_env: dict, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PIXABAY_API_KEY", "")
    result = runner.invoke(cli.app, ["images"])
    assert result.exit_code == 1
    assert "PIXABAY_API_KEY" in result.output


def test_cli_images_invalid_overrides(cli_env: dict):
    (cli_env["data"] / "overrides.yaml").write_text("knife: {}\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["images"])
    assert result.exit_code == 1
    assert "overrides.yaml" in result.output


def test_cli_images_api_error_keeps_progress(cli_env: dict):
    cli_env["fake"].results["ghost"] = []
    original = cli_env["fake"].handler

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("q") == "ghost":
            return httpx.Response(403, text="[ERROR 403] forbidden")
        return original(request)

    cli_env["fake"].handler = handler
    result = runner.invoke(cli.app, ["images"])
    assert result.exit_code == 1
    assert "vuelve a correr" in result.output
    cards = CARDS.validate_json(cli_env["cards_json"].read_bytes())
    assert cards[0].image.status == ImageStatus.FOUND


def test_cli_images_then_build_packs_image(cli_env: dict):
    assert runner.invoke(cli.app, ["images"]).exit_code == 0
    apkg = cli_env["data"] / "output" / "x.apkg"
    result = runner.invoke(cli.app, ["build"])
    assert result.exit_code == 0, result.output
    assert "Con imagen: 1" in result.output
    with zipfile.ZipFile(apkg) as zf:
        media = json.loads(zf.read("media"))
        [(idx, name)] = media.items()
        assert name.startswith("anki-series-") and name.endswith(".jpg")
        with Image.open(io.BytesIO(zf.read(idx))) as img:
            assert max(img.size) <= 400
