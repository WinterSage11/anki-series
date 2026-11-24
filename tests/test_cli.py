from pathlib import Path

from typer.testing import CliRunner

from anki_series import __version__
from anki_series.cli import app

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


def test_extract_without_dry_run_is_not_available_yet():
    result = runner.invoke(app, ["extract", str(FIXTURE), "--series", "X"])
    assert result.exit_code == 1
    assert "fase 2" in result.output


def test_extract_missing_path_fails_cleanly(tmp_path: Path):
    result = runner.invoke(
        app, ["extract", str(tmp_path / "nope.srt"), "--series", "X", "--dry-run"]
    )
    assert result.exit_code == 1
    assert "Error" in result.output
