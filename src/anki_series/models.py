"""Modelo de datos compartido por todas las etapas del pipeline.

Una ``Card`` nace en la fase de vocabulario (solo inglés) y las fases siguientes
van llenando campos (traducción, imagen, audio) sin cambiar el esquema.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

# ``pos`` de las tarjetas de expresiones multipalabra ("figure out", "make sense"). Las
# palabras sueltas usan las etiquetas universales de spaCy (NOUN, VERB...).
EXPRESSION_POS = frozenset({"phrasal", "idiom"})


class ImageStatus(StrEnum):
    PENDING = "pending"  # aún no se procesa
    FOUND = "found"  # imagen descargada y lista para empaquetar
    SKIPPED = "skipped"  # se decidió no poner imagen (palabra abstracta, override)
    NOT_FOUND = "not_found"  # se buscó pero no hubo resultado útil


class ImageInfo(BaseModel):
    status: ImageStatus = ImageStatus.PENDING
    query: str | None = None  # búsqueda en inglés, desambiguada por el LLM
    path: str | None = None  # ruta local del archivo ya redimensionado
    source: str | None = None  # p. ej. "pixabay"
    source_url: str | None = None
    credit: str | None = None  # solo metadata; no se muestra en la tarjeta
    reason: str | None = None  # por qué se omitió, si aplica


class AudioInfo(BaseModel):
    """Audios ``.mp3`` de la tarjeta (Fase 6). ``None`` = aún no se genera."""

    voice: str | None = None  # voz de edge-tts con la que se generaron
    word: str | None = None  # ruta local del .mp3 de la palabra (lema)
    example: str | None = None  # ruta local del .mp3 de la oración en inglés


class SourceRef(BaseModel):
    """De dónde salió la oración de ejemplo."""

    file: str  # nombre del .srt, p. ej. "S01E03.srt"
    start: str  # timestamp SRT, p. ej. "00:12:41,200"


class Sentence(BaseModel):
    """Oración limpia extraída de los subtítulos (salida de la fase 1)."""

    text: str
    source: SourceRef


class Card(BaseModel):
    series: str
    lemma: str
    pos: str  # etiqueta universal de spaCy (NOUN, VERB...) o phrasal/idiom (expresiones)
    rank: int = Field(ge=1)
    frequency: int = Field(ge=1)
    example_en: str
    source: SourceRef

    # Fase 4: traducción contextual
    translation: str | None = None
    example_es: str | None = None
    imageable: bool | None = None

    # Fase 5: imagen
    image: ImageInfo = Field(default_factory=ImageInfo)

    # Fase 6: audio (cards.json anteriores no traen este campo → vacío por default)
    audio: AudioInfo = Field(default_factory=AudioInfo)

    @property
    def key(self) -> str:
        """Identificador estable de la tarjeta (también base del GUID de Anki)."""
        return f"{self.series}|{self.lemma}|{self.pos}"
