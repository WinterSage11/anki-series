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
  deck.py       # (F3) genanki
  translate.py  # (F4) OpenAI Structured Outputs + caché
  images.py     # (F5) Pixabay + Pillow
tests/          # pytest; fixtures inventados en tests/fixtures/
data/           # NO versionado: raw/, cache/, output/, known/
```
Cada etapa lee/escribe JSON intermedio en `data/output/` (`words.json` → `cards.json` → `.apkg`) para poder inspeccionar y re-ejecutar cada fase por separado.

## Decisiones de diseño (no cambiar sin discutir)
1. **Unidad de tarjeta: lema + POS** (`run/VERB` y `run/NOUN` son tarjetas distintas). spaCy `en_core_web_sm`.
2. **Filtros**: excluir nombres propios (POS `PROPN` / NER `PERSON`), stopwords, números y puntuación. **Mantener slang y groserías.** Excluir lemas ya presentes en un mazo `.apkg` local (`--known-deck`): zip → `collection.anki21b` (zstd) / `.anki21` / `.anki2` (SQLite) → `notes.flds` separado por `\x1f` → campo configurable (default 0) → quitar HTML → lematizar. `wordfreq` solo como corte opcional.
3. **Entrada**: un `.srt` o una carpeta de `.srt` (temporada/serie). `--series` obligatorio; el GUID de la nota es `genanki.guid_for(series, lemma, pos)`.
4. **Traducción**: OpenAI con Structured Outputs usando modelos Pydantic. Modelo en `OPENAI_MODEL` (default `gpt-4o-mini`). Por lote se devuelve: `translation`, `example_es`, `imageable`, `image_query`. Español de México/LATAM. Cachear en `data/cache/` para no repetir llamadas.
5. **Tarjeta**: una sola dirección inglés → español. Frente: palabra + oración en inglés (+ imagen). Reverso: traducción + oración en español.
6. **Uso**: personal + amigos cercanos → **sin capturas de video**; solo imágenes con licencia redistribuible.
7. **Imágenes**: Pixabay principal. Créditos solo en metadata (no visibles en la tarjeta). El LLM decide `imageable` + `image_query`; `data/overrides.yaml` para ajustes manuales. Mejor sin imagen que con una engañosa. Redimensionar a máx. 400 px, JPG.
8. **genanki**: `model_id` y `deck_id` son constantes fijas en el código; nunca regenerarlas (romperían las actualizaciones en Anki).
9. El modelo `Card` incluye `image` desde el inicio; las fases posteriores solo llenan campos, no cambian el esquema.

## Fases
- [x] F0 Setup
- [ ] F1 Subtítulos (archivo o carpeta → oraciones limpias con timestamp)
- [ ] F2 Vocabulario (top 1,000 lema+POS, ejemplo, exclusión por mazo conocido) → `words.json`
- [ ] F3 Deck esqueleto solo inglés (campos de traducción/imagen vacíos)
- [ ] F4 Traducción contextual (OpenAI + caché)
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
