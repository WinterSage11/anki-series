"""Fase 1: leer subtítulos .srt y convertirlos en oraciones limpias con timestamp.

Flujo: bytes → texto (detección de encoding) → cues (bloques SRT) → fragmentos limpios
(sin etiquetas ni anotaciones) → oraciones (uniendo líneas/cues partidos y segmentando).
"""

import re
from dataclasses import dataclass
from pathlib import Path

from charset_normalizer import from_bytes

from anki_series.models import Sentence, SourceRef

# --- Expresiones regulares -------------------------------------------------------------

_TIMESTAMP = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
_TIMING_LINE = re.compile(rf"^\s*{_TIMESTAMP}\s*-->\s*{_TIMESTAMP}")

_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")  # <i>, </i>, <font color="...">
_ASS_TAG = re.compile(r"\{[^}]*\}")  # {\an8}, {\i1}
_ANNOTATION = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # [laughs], (door slams)
# Etiqueta de hablante al inicio: "MARCUS:", "MAN #2:", "DR. HALE:"
_SPEAKER = re.compile(r"^[A-Z][A-Z0-9 #.'\-]{0,30}:\s+")
_DIALOGUE_DASH = re.compile(r"^-+\s*")
_LEADING_ELLIPSIS = re.compile(r"^(\.{2,}|…)\s*")
_SPACES = re.compile(r"\s+")

# Créditos de quien hizo/sincronizó los subtítulos ("== sync, corrected by elderman ==",
# "Subtitles by ...", "www.addic7ed.com"). No son diálogo y meten palabras basura.
_CREDIT = re.compile(
    r"^\s*==.*==\s*$"  # == ... ==
    r"|\bsync(?:ed|hronized)?\b.*\bcorrected\b"  # "sync, corrected by", "synced and corrected"
    r"|\bcorrected\s+by\b"
    r"|\b(?:subtitles?|subs|captions?)\s+by\b"
    r"|open\s*subtitles|addic7ed"
    r"|\bwww\."
    r"|\b[\w-]+\.(?:com|org|net)\b",  # URLs sueltas: "tvsubtitles.net", "podnapisi.com"
    re.IGNORECASE,
)

# Fin de oración: . ! ? … seguidos (opcionalmente) de comillas de cierre.
_TERMINATED = re.compile(r"[.!?…][\"')]*$")
_SOFT_END = re.compile(r"(\.{2,}|…)[\"')]*$")  # puntos suspensivos: puede continuar
# Candidatos a corte: puntuación final + espacio + algo que pueda iniciar oración.
_SPLIT_POINT = re.compile(r"(?<=[.!?…])[\"')]*\s+(?=[\"'¿¡(]?[A-Z0-9])")

# Abreviaturas que terminan en punto pero no cierran oración.
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs.", "prof.", "lt.", "sgt."}

# Normalizar tipografía para que el lematizador vea siempre lo mismo ("don’t" == "don't").
_TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', " ": " "})


@dataclass(frozen=True)
class Cue:
    """Un bloque del .srt: timestamp de inicio/fin y sus líneas de texto crudas."""

    start: str
    end: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _Fragment:
    """Texto limpio de un cue (o de un turno de diálogo dentro de un cue)."""

    text: str
    start: str
    new_speaker: bool  # empezó con guion de diálogo: no se une con lo anterior


# --- Lectura ---------------------------------------------------------------------------


def decode_bytes(raw: bytes) -> str:
    """Decodifica bytes de un .srt.

    Primero intenta UTF-8 (con o sin BOM), que es lo más común y no es ambiguo. Solo si
    falla recurre a charset-normalizer, que adivina por estadística (cp1252, latin-1...).
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    results = from_bytes(raw)
    best = results.best()
    if best is None:
        return raw.decode("utf-8", errors="replace")
    # En textos cortos cp1250/cp1252 suelen empatar (mismo "caos"). Como los subtítulos
    # son en inglés, ante un empate preferimos cp1252 (Windows-1252, el más común).
    for match in results:
        if "cp1252" in match.could_be_from_charset and match.percent_chaos <= best.percent_chaos:
            return raw.decode("cp1252")
    return str(best)


def read_srt(path: Path) -> str:
    return decode_bytes(path.read_bytes())


def find_srt_files(path: Path) -> list[Path]:
    """Un .srt → [ese archivo]; una carpeta → todos sus .srt (recursivo, ordenados)."""
    if path.is_file():
        if path.suffix.lower() != ".srt":
            raise ValueError(f"No es un archivo .srt: {path}")
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".srt")
        if not files:
            raise ValueError(f"La carpeta no contiene archivos .srt: {path}")
        return files
    raise FileNotFoundError(path)


# --- Parseo ----------------------------------------------------------------------------


def _format_timestamp(h: str, m: str, s: str, ms: str) -> str:
    return f"{int(h):02d}:{m}:{s},{ms.ljust(3, '0')}"


def parse_srt(text: str) -> list[Cue]:
    """Convierte el texto de un .srt en cues. Tolera CRLF, índices faltantes y basura."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip("\n").split("\n")
        for i, line in enumerate(lines):
            match = _TIMING_LINE.match(line)
            if match:
                g = match.groups()
                cues.append(
                    Cue(
                        start=_format_timestamp(*g[:4]),
                        end=_format_timestamp(*g[4:]),
                        lines=tuple(lines[i + 1 :]),
                    )
                )
                break
    return cues


# --- Limpieza --------------------------------------------------------------------------


def is_credit_line(line: str) -> bool:
    """¿La línea es un crédito de los subtítulos (sync/corrección, sitio web)?"""
    return bool(_CREDIT.search(line))


def clean_line(line: str) -> str:
    """Quita etiquetas, anotaciones y nombres de hablante de una línea.

    Devuelve "" si la línea no tiene diálogo (música, efectos de sonido, créditos de los
    subtítulos). Conserva el guion inicial de diálogo para que el llamador lo detecte.
    """
    line = line.translate(_TYPOGRAPHY)
    if "♪" in line or "♫" in line:  # letras de canciones: no son diálogo
        return ""
    line = _HTML_TAG.sub("", line)
    line = _ASS_TAG.sub("", line)
    if is_credit_line(line):
        return ""
    line = _ANNOTATION.sub("", line)
    line = _SPACES.sub(" ", line).strip()

    dash = _DIALOGUE_DASH.match(line)
    prefix = "- " if dash else ""
    body = line[dash.end() :] if dash else line
    body = _SPEAKER.sub("", body).strip()
    if not any(ch.isalpha() for ch in body):
        return ""
    return prefix + body


def _cue_fragments(cue: Cue) -> list[_Fragment]:
    """Une las líneas partidas de un cue; separa turnos marcados con guion."""
    # Un crédito puede venir partido en varias líneas ("== sync, corrected by" / "elderman
    # =="): se revisa el cue completo además de cada línea (en ``clean_line``).
    whole = " ".join(_ASS_TAG.sub("", _HTML_TAG.sub("", raw)).strip() for raw in cue.lines)
    if is_credit_line(whole):
        return []
    fragments: list[_Fragment] = []
    current: list[str] = []
    current_is_turn = False

    def flush() -> None:
        if current:
            fragments.append(_Fragment(" ".join(current), cue.start, current_is_turn))

    for raw in cue.lines:
        line = clean_line(raw)
        if not line:
            continue
        if line.startswith("- "):
            flush()
            current, current_is_turn = [line[2:]], True
        else:
            current.append(line)
    flush()
    return fragments


# --- Segmentación ----------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Parte un texto en oraciones respetando abreviaturas comunes (Mr., Dr., ...)."""
    pieces: list[str] = []
    last = 0
    for match in _SPLIT_POINT.finditer(text):
        candidate = text[last : match.start()].strip()
        last_word = candidate.rsplit(" ", 1)[-1].lower()
        if last_word in _ABBREVIATIONS:
            continue
        pieces.append(candidate)
        last = match.end()
    tail = text[last:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def _join(left: str, right: str) -> str:
    left = _SOFT_END.sub("", left) if _SOFT_END.search(left) else left
    return f"{left.rstrip()} {right}".strip()


def cues_to_sentences(cues: list[Cue], file: str) -> list[Sentence]:
    """Arma oraciones completas aunque crucen varios cues.

    Regla de unión entre cues: si lo acumulado no termina en puntuación final, se une con
    lo siguiente. Si termina en "..." solo se une cuando lo siguiente arranca con "..." o
    minúscula ("I was going to..." + "...call you"). Un guion de diálogo siempre corta.
    """
    sentences: list[Sentence] = []
    pending = ""
    pending_start = ""

    def emit(text: str, start: str) -> None:
        if any(ch.isalpha() for ch in text):
            sentences.append(Sentence(text=text, source=SourceRef(file=file, start=start)))

    for cue in cues:
        for frag in _cue_fragments(cue):
            text = frag.text
            continues = bool(_LEADING_ELLIPSIS.match(text)) or text[:1].islower()
            text = _LEADING_ELLIPSIS.sub("", text)

            if pending:
                soft = bool(_SOFT_END.search(pending))
                if frag.new_speaker or (soft and not continues):
                    emit(pending, pending_start)
                    pending = ""
                elif soft or not _TERMINATED.search(pending):
                    text = _join(pending, text)

            start = pending_start if pending else frag.start
            pending = ""
            pieces = split_sentences(text)
            for i, piece in enumerate(pieces):
                piece_start = start if i == 0 else frag.start
                is_last = i == len(pieces) - 1
                if is_last and (not _TERMINATED.search(piece) or _SOFT_END.search(piece)):
                    pending, pending_start = piece, piece_start
                else:
                    emit(piece, piece_start)

    if pending:
        emit(pending, pending_start)
    return sentences


# --- API pública -----------------------------------------------------------------------


def load_sentences(path: Path) -> list[Sentence]:
    """Lee un .srt o una carpeta de .srt y devuelve todas las oraciones limpias.

    ``SourceRef.file`` es la ruta relativa a la carpeta de entrada (o el nombre del
    archivo si la entrada es un solo .srt), p. ej. ``"S01/S01E03.srt"``.
    """
    base = path if path.is_dir() else path.parent
    sentences: list[Sentence] = []
    for srt in find_srt_files(path):
        cues = parse_srt(read_srt(srt))
        sentences.extend(cues_to_sentences(cues, srt.relative_to(base).as_posix()))
    return sentences
