"""Genera .apkg mínimos (inventados) para tests: colección SQLite dentro de un zip."""

import json
import sqlite3
import zipfile
from pathlib import Path

import zstandard

FIELDS = ["Índice", "Palabra", "Definición"]
NOTES = [
    ["1", "have", "(v.) Tener."],
    ["2", "<b>store</b>&nbsp;", "(sus.) Tienda"],
    ["3", "left [sound:left.mp3]", "(adj.) Izquierda"],
    ["4", "give up", "(v.) Rendirse"],
]


def _notes_db(path: Path, *, new_schema: bool, notes=NOTES) -> None:
    """Colección mínima de Anki: tabla notes + definición de campos (esquema viejo o nuevo)."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, mid INTEGER, flds TEXT)")
    if new_schema:
        db.execute("CREATE TABLE fields (ntid INTEGER, ord INTEGER, name TEXT)")
        db.executemany("INSERT INTO fields VALUES (7, ?, ?)", list(enumerate(FIELDS)))
    else:
        db.execute("CREATE TABLE col (models TEXT)")
        models = {"7": {"name": "Test", "flds": [{"name": n} for n in FIELDS]}}
        db.execute("INSERT INTO col VALUES (?)", (json.dumps(models),))
    db.executemany(
        "INSERT INTO notes (mid, flds) VALUES (7, ?)", [("\x1f".join(n),) for n in notes]
    )
    db.commit()
    db.close()


def make_apkg(tmp_path: Path, fmt: str = "anki21", notes=NOTES) -> Path:
    db_path = tmp_path / "col.sqlite"
    _notes_db(db_path, new_schema=fmt == "anki21b", notes=notes)
    stub = tmp_path / "stub.sqlite"
    _notes_db(stub, new_schema=False, notes=[["Please update to the latest Anki.", "", ""]])
    apkg = tmp_path / f"deck-{fmt}.apkg"
    with zipfile.ZipFile(apkg, "w") as zf:
        data = db_path.read_bytes()
        if fmt == "anki21b":
            data = zstandard.ZstdCompressor().compress(data)
        zf.writestr(f"collection.{fmt}", data)
        if fmt != "anki2":
            zf.writestr("collection.anki2", stub.read_bytes())  # stub, como Anki real
        zf.writestr("media", "{}")
    return apkg
