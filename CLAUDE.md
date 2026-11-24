# CLAUDE.md — anki-series

## Objetivo
CLI en Python que convierte subtítulos `.srt` en inglés de una serie en un mazo `.apkg` de Anki: top 1,000 lemas (lema + POS) con una oración real de la serie, traducción contextual a español de México/LATAM e imagen cuando aporta valor (empaquetada en el `.apkg`).

## Comandos
```bash
uv sync                       # instalar dependencias
uv run pytest                 # tests (obligatorios en verde antes de cada commit)
uv run ruff check .           # lint
uv run ruff format .          # formato
uv run anki-series --help     # CLI
```
- Usar siempre `uv add <pkg>` / `uv add --dev <pkg>`; nunca `pip install` ni instalaciones globales.
- Agregar cada dependencia en la fase que la usa, no antes.

## Estructura
```
src/anki_series/
  models.py     # Card, ImageInfo (pydantic). Contrato de datos entre etapas
  config.py     # Settings (pydantic-settings, lee .env)
  cli.py        # typer: extract / translate / images / build / all
  subtitles.py  # (F1) leer y limpiar .srt (archivo o carpeta)
  vocab.py      # (F2) lematizar, filtrar, contar, elegir ejemplo, excluir mazo conocido
  deck.py       # (F3) genanki: modelo de nota fijo, GUID estable, resaltado, media
  translate.py  # (F4) OpenAI Structured Outputs + caché
  images.py     # (F5) Pixabay + Pillow
tests/          # pytest; fixtures inventados en tests/fixtures/, .apkg sintéticos con tests/apkg_factory.py
data/           # NO versionado: raw/, cache/, output/, known/
```
Cada etapa lee/escribe JSON intermedio en `data/output/` (`words.json` → `cards.json` → `.apkg`) para poder inspeccionar y re-ejecutar cada fase por separado.

## Decisiones de diseño (no cambiar sin discutir)
1. **Unidad de tarjeta: lema + POS** (`run/VERB` y `run/NOUN` son tarjetas distintas). spaCy `en_core_web_sm`.
2. **Filtros**: excluir nombres propios (POS `PROPN` / NER `PERSON`), stopwords, números y puntuación. **Mantener slang y groserías.** La stoplist de spaCy solo se aplica a tokens que no son `NOUN`/`VERB`/`ADJ` (incluye `get`, `go`, `make`, `say`, `front`… que sí son vocabulario). Excluir lemas ya presentes en un mazo `.apkg` local (`--known-deck`): zip → `collection.anki21b` (zstd) > `.anki21` > `.anki2` (SQLite; en orden de prioridad, porque si existe uno nuevo el `.anki2` es un stub vacío) → `notes.flds` separado por `\x1f` → campo configurable por índice o nombre (`--known-field` / `KNOWN_DECK_FIELD`, default 0; error con la lista de campos si no contiene palabras) → quitar HTML y `[sound:]` → lematizar. Se ignoran entradas de varias palabras (phrasal verbs). La exclusión es por lema, para todos sus POS, y ocurre antes del corte top N. `wordfreq` solo como corte opcional (aún no implementado).
3. **Entrada**: un `.srt` o una carpeta de `.srt` (temporada/serie). `--series` obligatorio; el GUID de la nota es `genanki.guid_for(series, lemma, pos)`.
4. **Traducción**: OpenAI con Structured Outputs usando modelos Pydantic. Modelo en `OPENAI_MODEL` (default `gpt-4o`). Por lote se devuelve: `translation`, `example_es`, `imageable`, `image_query`. Español de México/LATAM. Cachear en `data/cache/` para no repetir llamadas.
5. **Tarjeta**: una sola dirección inglés → español. Frente: palabra + oración en inglés (+ imagen). Reverso: traducción + oración en español.
6. **Uso**: personal + amigos cercanos → **sin capturas de video**; solo imágenes con licencia redistribuible.
7. **Imágenes**: Pixabay principal. Créditos solo en metadata (no visibles en la tarjeta). El LLM decide `imageable` + `image_query`; `data/overrides.yaml` para ajustes manuales. Mejor sin imagen que con una engañosa. Redimensionar a máx. 400 px, JPG.
8. **genanki**: `MODEL_ID` es una constante global fija en `deck.py`; nunca regenerarla (rompería las actualizaciones en Anki). `deck_id` es determinista por serie: `2**30 + sha256("anki-series|<serie>")[:8] % 2**30` (series distintas no se mezclan al importar); mazo `anki-series::<serie>`. Campos del modelo (no cambiar orden ni nombres): `Word`, `POS`, `ExampleEN` (HTML escapado, palabra en `<b>` re-lematizando la oración con spaCy), `Translation`, `ExampleES`, `Image` (`<img>`), `AudioWord`, `AudioExample` (`[sound:…mp3]`, vacíos hasta tener voces neuronales; **sin TTS nativo**, sonaba mal), `Credits` y `Source` (metadata oculta). Plantilla: tarjeta blanca centrada, secciones separadas por líneas. Frente = `Word` + `POS` + `{{#AudioWord}}` arriba a la derecha. Reverso = (1) `Word` + audio, (2) `traducción:`, (3) `{{#Image}}` centrada, (4) `ejemplo:` con `ExampleEN` + `{{#AudioExample}}` a la derecha y `ExampleES` en cursiva. Media renombrada a `anki-series-<sha1(key)[:12]>.<ext>` (la carpeta de media de Anki es global). Tag de la nota = POS.
9. El modelo `Card` incluye `image` desde el inicio; las fases posteriores solo llenan campos, no cambian el esquema.

## Fases
- [x] F0 Setup
- [x] F1 Subtítulos (archivo o carpeta → oraciones limpias con timestamp)
- [x] F2 Vocabulario (top 1,000 lema+POS, ejemplo, exclusión por mazo conocido) → `words.json` (lista de `Card` solo inglés). Ejemplo = oración de 4–20 palabras más cercana a 10; empate → primera aparición.
- [x] F3 Deck esqueleto (`build`: `cards.json` o `words.json` → `<serie>.apkg`; campos de traducción/imagen vacíos hasta F4/F5)
- [x] F4 Traducción contextual (`translate`: `words.json` → `cards.json`). `chat.completions.parse` con `TranslationBatch` (Pydantic); lotes de 20 (`--batch-size`), `--limit N` escribe solo las primeras N. Caché en `data/cache/translations.json` (clave = sha1 de versión de prompt + modelo + lema + POS + oración; no incluye serie), guardado atómico tras cada lote. Reintentos propios con backoff 2/4/8 s ante conexión/timeout/429/5xx (cliente con `max_retries=0`); otros errores fallan de inmediato. `image_query` va a `card.image.query` (el estado de la imagen lo decide F5); si `imageable` sin query → `imageable=False`. Subir `PROMPT_VERSION` al cambiar prompt/esquema. Phrasal verbs: el esquema pide `phrase` y `phrase_translation` **antes** de `translation` (los campos se generan en orden) para que `translation` sea la palabra sola (run → correr aunque la oración diga "run out of"); esos dos campos solo viven en el caché. `gpt-4o-mini` no respeta bien esta regla (por eso el default es `gpt-4o`; `gpt-4.1-mini` también funciona). `.env` lo lee pydantic-settings (sin `python-dotenv`).
- [ ] F5 Imágenes (Pixabay)
- [ ] F6 Opcional (TTS, overrides avanzados…)

## Convenciones
- Python ≥ 3.12, type hints en funciones públicas, `ruff` (line-length 100).
- Código, identificadores y commits en inglés; docs (README, CLAUDE.md) en español.
- **Conventional Commits**: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`.
- Una rama por fase (`feat/phase-1-subtitles`) con PR hacia `main`. Repo GitHub privado.
- No hacer commits ni push sin confirmación explícita del usuario.
- Nunca versionar `.env`, `data/`, `*.apkg` ni subtítulos reales. Los fixtures de tests son inventados.
- Tests sin red: mockear OpenAI y Pixabay.
