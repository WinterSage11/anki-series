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
  cli.py        # typer: extract / translate / images / audio / build / all
  subtitles.py  # (F1) leer y limpiar .srt (archivo o carpeta)
  vocab.py      # (F2) lematizar, filtrar, contar, elegir ejemplo, excluir mazo conocido
  deck.py       # (F3) genanki: modelo de nota fijo, GUID estable, resaltado, media
  translate.py  # (F4) OpenAI Structured Outputs + caché
  images.py     # (F5) Pixabay + Pillow
  audio.py      # (F6) edge-tts (voces neuronales) + caché
tests/          # pytest; fixtures inventados en tests/fixtures/, .apkg sintéticos con tests/apkg_factory.py
data/           # NO versionado: raw/, cache/, output/, known/
```
Cada etapa lee/escribe JSON intermedio en `data/output/` (`words.json` → `cards.json` → `.apkg`) para poder inspeccionar y re-ejecutar cada fase por separado.

## Decisiones de diseño (no cambiar sin discutir)
1. **Unidad de tarjeta: lema + POS** (`run/VERB` y `run/NOUN` son tarjetas distintas). spaCy `en_core_web_sm`.
2. **Filtros**: excluir nombres propios (POS `PROPN` / NER `PERSON`), interjecciones (`INTJ`: yeah, oh, okay, hey…), muletillas vocales aunque spaCy las etiquete `NOUN`/`ADJ` (`hmm`, `mm`, `mhm`, `uh`, `um`, `huh`, `ah`, `er` y alargadas como `hmmm`; no `err`), stopwords, números y puntuación. **Mantener slang y groserías.** Nombres colados: si más de la mitad de las apariciones de un lema en el corpus son `PROPN`/`PERSON`, se excluye en todos sus POS (`harvey/NOUN`); costo aceptado: `mom`/`dad` en vocativo podrían caer. Contracciones informales (`gonna`, `gotta`, `wanna`, como token único o partidas por spaCy en `gon`+`na` sin espacio) cuentan como `go/VERB`, `get/VERB`, `want/VERB` (`vocab.token_lemma_pos`, también lo usa el resaltado de `deck.py`). `--exclude-common N` (default 0) excluye las N palabras más frecuentes del inglés según `wordfreq` (solo palabras alfabéticas, comparadas contra el lema sin lematizar la lista). La stoplist de spaCy solo se aplica a tokens que no son `NOUN`/`VERB`/`ADJ` (incluye `get`, `go`, `make`, `say`, `front`… que sí son vocabulario). Excluir lemas ya presentes en un mazo `.apkg` local (`--known-deck`): zip → `collection.anki21b` (zstd) > `.anki21` > `.anki2` (SQLite; en orden de prioridad, porque si existe uno nuevo el `.anki2` es un stub vacío) → `notes.flds` separado por `\x1f` → campo configurable por índice o nombre (`--known-field` / `KNOWN_DECK_FIELD`, default 0; error con la lista de campos si no contiene palabras) → quitar HTML y `[sound:]` → lematizar. Se ignoran entradas de varias palabras (phrasal verbs). La exclusión (mazo conocido y `wordfreq`) es por lema, para todos sus POS, y ocurre antes del corte top N.
3. **Entrada**: un `.srt` o una carpeta de `.srt` (temporada/serie). `--series` obligatorio; el GUID de la nota es `genanki.guid_for(series, lemma, pos)`.
4. **Traducción**: OpenAI con Structured Outputs usando modelos Pydantic. Modelo en `OPENAI_MODEL` (default `gpt-4o`). Por lote se devuelve: `translation`, `example_es`, `imageable`, `image_query`. Español de México/LATAM. Cachear en `data/cache/` para no repetir llamadas.
5. **Tarjeta**: una sola dirección inglés → español. Frente: palabra + oración en inglés (+ imagen). Reverso: traducción + oración en español.
6. **Uso**: personal + amigos cercanos → **sin capturas de video**; solo imágenes con licencia redistribuible.
7. **Imágenes**: Pixabay principal. Créditos solo en metadata (no visibles en la tarjeta). El LLM decide `imageable` + `image_query`; `data/overrides.yaml` para ajustes manuales. Mejor sin imagen que con una engañosa. Redimensionar a máx. 400 px, JPG.
8. **genanki**: `MODEL_ID` es una constante global fija en `deck.py`; nunca regenerarla (rompería las actualizaciones en Anki). `deck_id` es determinista por serie: `2**30 + sha256("anki-series|<serie>")[:8] % 2**30` (series distintas no se mezclan al importar); mazo `anki-series::<serie>`. Campos del modelo (no cambiar orden ni nombres): `Word`, `POS`, `ExampleEN` (HTML escapado, palabra en `<b>` re-lematizando la oración con spaCy), `Translation`, `ExampleES`, `Image` (`<img>`), `AudioWord`, `AudioExample` (`[sound:…mp3]` generados por F6 con edge-tts; vacíos si no se corrió `audio`; **sin TTS nativo**, sonaba mal), `Credits` y `Source` (metadata oculta). Plantilla: tarjeta blanca centrada, secciones separadas por líneas. Frente = `Word` + `POS` + `{{#AudioWord}}` arriba a la derecha. Reverso = (1) `Word` + audio, (2) `traducción:`, (3) `{{#Image}}` centrada, (4) `ejemplo:` con `ExampleEN` + `{{#AudioExample}}` a la derecha y `ExampleES` en cursiva. Media renombrada a `anki-series-<sha1(key)[:12]>.<ext>` (la carpeta de media de Anki es global): imágenes con key = `card.key`; audios con key = `audio|<nombre del mp3 en caché>` (identifica voz + texto: se comparte entre tarjetas y cambia si cambia la voz o el texto, porque Anki no reemplaza media existente al reimportar). Tag de la nota = POS.
9. El modelo `Card` incluye `image` desde el inicio; las fases posteriores solo llenan campos, no cambian el esquema.

## Fases
- [x] F0 Setup
- [x] F1 Subtítulos (archivo o carpeta → oraciones limpias con timestamp). Se descartan créditos de subtítulos (`subtitles.is_credit_line`: `== … ==`, `sync/synced … corrected`, `corrected by`, `subtitles/subs/captions by`, `opensubtitles`, `addic7ed`, `www.`, dominios `.com/.org/.net`), por línea y por cue completo (un crédito puede venir partido en dos líneas).
- [x] F2 Vocabulario (top 1,000 lema+POS, ejemplo, exclusión por mazo conocido) → `words.json` (lista de `Card` solo inglés). Ejemplo = oración de 4–20 palabras más cercana a 10; empate → primera aparición.
- [x] F3 Deck esqueleto (`build`: `cards.json` o `words.json` → `<serie>.apkg`; campos de traducción/imagen vacíos hasta F4/F5)
- [x] F4 Traducción contextual (`translate`: `words.json` → `cards.json`). `chat.completions.parse` con `TranslationBatch` (Pydantic); lotes de 20 (`--batch-size`), `--limit N` escribe solo las primeras N. Caché en `data/cache/translations.json` (clave = sha1 de versión de prompt + modelo + lema + POS + oración; no incluye serie), guardado atómico tras cada lote. Reintentos propios con backoff 2/4/8 s ante conexión/timeout/429/5xx (cliente con `max_retries=0`); otros errores fallan de inmediato. `image_query` va a `card.image.query` (el estado de la imagen lo decide F5); si `imageable` sin query → `imageable=False`. Subir `PROMPT_VERSION` al cambiar prompt/esquema. Phrasal verbs: el esquema pide `phrase` y `phrase_translation` **antes** de `translation` (los campos se generan en orden) para que `translation` sea la palabra sola (run → correr aunque la oración diga "run out of"); esos dos campos solo viven en el caché. `gpt-4o-mini` no respeta bien esta regla (por eso el default es `gpt-4o`; `gpt-4.1-mini` también funciona). `.env` lo lee pydantic-settings (sin `python-dotenv`).
- [x] F5 Imágenes (`images`: `cards.json` → `cards.json` actualizado in place por default; `-i`/`-o`, `--limit N` procesa N pero escribe todas, `--overrides`). Se procesan tarjetas `pending` ya traducidas: `imageable=False` → `SKIPPED` ("no ilustrable"); `imageable=None` no se toca. Pixabay (`httpx`, `image_type=photo`, `safesearch`, 5 hits): un hit es relevante si comparte una palabra (≥3 letras, por prefijo) de la query con sus `tags`; sin relevantes o descargas fallidas → `NOT_FOUND`. JPEG máx. 400 px (lado largo, Pillow, q=80 optimize). Caché en `data/cache/images/<sha1(query)>.jpg` + `index.json` (también guarda `found: false`; borrar la entrada para reintentar), guardado atómico tras cada búsqueda; `cards.json` checkpoint atómico cada 10 tarjetas y ante error/Ctrl+C. Reintentos 2/4/8 s ante red/timeout/429 (respeta `X-RateLimit-Reset`)/5xx; otros errores fallan sin mostrar la API key. `data/overrides.yaml`: llave `lema` o `lema/POS` (la más específica gana), valor con exactamente uno de `query`, `image` (ruta local, relativa al proyecto) o `skip: true`; los overrides se aplican siempre (aunque la tarjeta ya no esté `pending` o no sea ilustrable). Créditos: `"<user> / Pixabay"` + `pageURL`. Tests: `httpx.MockTransport`; el CLI obtiene el cliente HTTP de `cli._pixabay_http()`.
- [x] F6 Audio (`audio`: `cards.json` → `cards.json` actualizado in place por default; `-i`/`-o`, `--limit N` procesa N pero escribe todas, `--voice` (default `TTS_VOICE` = `en-US-AndrewNeural`), `--concurrency` (default 4)). `edge-tts` (voces neuronales de Microsoft, sin API key). Por tarjeta dos `.mp3`: lema y `example_en` en texto plano (sin HTML, entidades resueltas). `Card.audio: AudioInfo` (`voice`, `word`, `example`; default vacío → `cards.json` viejos siguen cargando). Una tarjeta se procesa si su `audio` no coincide con las rutas esperadas para (voz, texto) o falta el archivo. Caché = el archivo mismo: `data/cache/audio/<sha1(voz\x1ftexto)>.mp3` (borrarlo para regenerar); escritura atómica vía `.tmp`; un solo trabajo por archivo aunque varias tarjetas compartan texto. Concurrencia con `asyncio.Semaphore`. Reintentos 2/4/8 s ante `EdgeTTSException`/`aiohttp.ClientError`/timeout/conexión o archivo vacío; otros errores (p. ej. `ValueError` por voz inválida) fallan de inmediato con `AudioError`. Checkpoint atómico de `cards.json` cada 10 tarjetas y ante error/Ctrl+C (se cancela lo pendiente). Tests: sintetizador falso inyectado o `edge_tts.Communicate` reemplazado con monkeypatch.
- [x] F7 Pulido (detectado con una temporada real): sin `INTJ`, voto de nombres propios por lema, contracciones `gonna`/`gotta`/`wanna` → verbo, `extract --exclude-common N` (`wordfreq`). Comando `all`: llama a las funciones de los comandos `extract → translate → images → audio → build` con sus rutas por default en `data/output/` (reutiliza cachés y checkpoints; se detiene en la primera etapa que falla). Acepta `input_path`, `--series`, `--top`, `--known-deck`, `--known-field`, `--exclude-common`, `--limit` (solo se pasa a `translate`: `cards.json` queda con N tarjetas), `--model`, `--voice`, `--overrides`, `--concurrency`. Tests de `all` reutilizan los dobles `FakeClient`, `FakePixabay` y `FakeCommunicate` de los tests de cada fase.
- [ ] Ideas futuras: overrides avanzados (traducción/ejemplo manual), wordfreq por lema.

## Convenciones
- Python ≥ 3.12, type hints en funciones públicas, `ruff` (line-length 100).
- Código, identificadores y commits en inglés; docs (README, CLAUDE.md) en español.
- **Conventional Commits**: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`.
- Una rama por fase (`feat/phase-1-subtitles`) con PR hacia `main`. Repo GitHub privado.
- No hacer commits ni push sin confirmación explícita del usuario.
- Nunca versionar `.env`, `data/`, `*.apkg` ni subtítulos reales. Los fixtures de tests son inventados.
- Tests sin red: mockear OpenAI y Pixabay.
