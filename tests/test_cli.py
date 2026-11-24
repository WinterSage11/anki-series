from typer.testing import CliRunner

from anki_series import __version__
from anki_series.cli import app


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
