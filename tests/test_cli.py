from pathlib import Path

from apkg_factory import make_apkg
from pydantic import TypeAdapter
from typer.testing import CliRunner

from anki_series import __version__
from anki_series.cli import app
from anki_series.models import Card

FIXTURE = Path(__file__).parent / "fixtures" / "sample.srt"
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


def test_extract_missing_path_fails_cleanly(tmp_path: Path):
    result = runner.invoke(
        app, ["extract", str(tmp_path / "nope.srt"), "--series", "X", "--dry-run"]
    )
    assert result.exit_code == 1
    assert "Error" in result.output
