# anki-series

Herramienta de línea de comandos que toma los subtítulos (`.srt`) en inglés de una serie y genera un mazo de Anki (`.apkg`) con:

- Las 1,000 palabras más usadas (lema + categoría gramatical).
- Una oración real de la serie como ejemplo.
- Traducción al español de México/LATAM considerando el contexto de la oración.
- Una imagen cuando aporta valor (empaquetada dentro del `.apkg`).
- Audio con voz neuronal de la palabra y de la oración (empaquetado dentro del `.apkg`).

> Estado: **Fase 7** (vocabulario → `words.json` → traducción con OpenAI → `cards.json` → imágenes de Pixabay → audio con edge-tts → mazo `.apkg`, todo con un solo comando `all`). Ver el plan de fases en [CLAUDE.md](CLAUDE.md).

## Requisitos

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 (uv lo instala si no lo tienes)

## Instalación

```bash
uv sync
cp .env.example .env   # y llena OPENAI_API_KEY y PIXABAY_API_KEY
```

## Uso rápido: todo en un comando

```bash
# Prueba barata: solo 20 palabras en todo el flujo
uv run anki-series all data/raw/mi-serie/ --series "Mi Serie" --exclude-common 500 --limit 20

# Completo: un archivo o una carpeta con todos los .srt de la temporada/serie
uv run anki-series all data/raw/mi-serie/ --series "Mi Serie" \
    --known-deck data/known/mi-mazo.apkg --known-field Palabra --exclude-common 500
# → data/output/mi-serie.apkg
```

`all` corre `extract → translate → images → audio → build` con las rutas por default de `data/output/`. Cada etapa usa su caché, así que si algo falla (red, API key, límite de uso) se detiene ahí: corrige y vuelve a correr el mismo comando; lo ya hecho no se repite ni se vuelve a cobrar.

| Opción | Default | Qué hace |
|---|---|---|
| `--series` | — (obligatorio) | Nombre de la serie (define el mazo y el GUID de las notas) |
| `--top` | 1000 | Cuántos lemas conservar |
| `--known-deck` / `--known-field` | — / `KNOWN_DECK_FIELD` o `0` | Mazo con palabras que ya conoces |
| `--exclude-common` | 0 | Excluye las N palabras más frecuentes del inglés (`wordfreq`) |
| `--limit` | — | Solo las primeras N palabras en todo el flujo (para probar) |
| `--model` | `OPENAI_MODEL` | Modelo de OpenAI |
| `--overrides` | `data/overrides.yaml` | Ajustes manuales de imágenes |
| `--voice` | `TTS_VOICE` | Voz de edge-tts |
| `--concurrency` | 4 | Audios generados en paralelo |

## Etapas por separado

Útil para inspeccionar o re-ejecutar una sola fase:

```bash
uv run anki-series version

# Vista previa de las oraciones limpias que se extraen (no escribe nada)
uv run anki-series extract data/raw/mi-serie/ --series "Mi Serie" --dry-run --limit 20

# Vocabulario: top 1,000 lemas (lema + categoría) con oración de ejemplo
# → data/output/words.json
uv run anki-series extract data/raw/mi-serie/ --series "Mi Serie" \
    --known-deck data/known/mi-mazo.apkg --known-field Palabra
```

Opciones de `extract`:

| Opción | Default | Qué hace |
|---|---|---|
| `--top` | 1000 | Cuántos lemas conservar |
| `--known-deck` | — | Mazo `.apkg` con palabras que ya conoces; se excluyen sus lemas |
| `--known-field` | `KNOWN_DECK_FIELD` o `0` | Campo del mazo con la palabra en inglés (índice o nombre). Si el campo no contiene palabras, el error lista los campos disponibles |
| `--exclude-common` | 0 | Excluye las N palabras más frecuentes del inglés según [wordfreq](https://github.com/rspeer/wordfreq) (p. ej. 500 quita `get`, `go`, `know`, `time`…). 0 = no excluir |
| `-o/--output` | `data/output/words.json` | Ruta del JSON de salida |
| `--limit` | 10 | Filas de muestra en pantalla |

Para no repetir `--known-field`, ponlo en `.env` (p. ej. `KNOWN_DECK_FIELD=Palabra`).

Qué se filtra automáticamente:

- Créditos de quien hizo los subtítulos (`== sync, corrected by … ==`, `Subtitles by …`, OpenSubtitles, Addic7ed, URLs).
- Nombres propios, números, puntuación, stopwords (pronombres, preposiciones…), interjecciones (`yeah`, `oh`, `okay`, `hey`…) y muletillas (`hmm`, `mm-hmm`, `uh`, `um`…). El slang y las groserías se quedan.
- Nombres de personajes que spaCy a veces etiqueta como sustantivo (`Harvey, you gotta…`): si un lema aparece como nombre propio en más de la mitad de sus apariciones, se descarta. Efecto secundario posible: `mom`/`dad` usados como vocativo.
- `gonna`, `gotta` y `wanna` cuentan como `go`, `get` y `want` (verbo), y en la tarjeta se resalta la contracción completa.

### Traducir (OpenAI)

Requiere `OPENAI_API_KEY` en `.env` (y opcionalmente `OPENAI_MODEL`, default `gpt-4o`).

```bash
# Prueba barata con 10 tarjetas → data/output/cards.json (solo esas 10)
uv run anki-series translate --limit 10

# Todo words.json → data/output/cards.json
uv run anki-series translate
```

Por cada palabra el modelo devuelve, en español de México/LATAM y considerando la oración de la serie: la traducción de la palabra, la traducción natural de la oración, si la palabra es ilustrable con una foto y la búsqueda en inglés para la imagen (Fase 5).

| Opción | Default | Qué hace |
|---|---|---|
| `-i/--input` | `data/output/words.json` | JSON de entrada |
| `-o/--output` | `data/output/cards.json` | JSON de salida (el que usa `build`) |
| `--limit` | — | Solo las primeras N tarjetas (y escribe solo esas) |
| `--batch-size` | 20 | Tarjetas por llamada a la API |
| `--model` | `OPENAI_MODEL` | Modelo a usar en esta corrida |

**Caché**: cada lote se guarda en `data/cache/translations.json` al terminar. Si el proceso se interrumpe o falla, vuelve a correr el mismo comando: solo se piden los lotes que faltan. Cambiar el modelo o la oración de ejemplo genera una traducción nueva. Los errores transitorios (red, timeout, límite de uso, 5xx) se reintentan hasta 3 veces con espera creciente.

### Imágenes (Pixabay)

Requiere `PIXABAY_API_KEY` en `.env` ([obtenla aquí](https://pixabay.com/api/docs/)) y haber corrido `translate` (el modelo decide qué palabras son ilustrables y con qué búsqueda).

```bash
# Prueba con 10 tarjetas (actualiza data/output/cards.json; las demás quedan pendientes)
uv run anki-series images --limit 10

# Todas las pendientes
uv run anki-series images
```

- Solo se buscan las palabras concretas (`imageable`); las demás quedan "omitidas".
- Si Pixabay no devuelve un resultado cuyos tags coincidan con la búsqueda, la tarjeta queda **sin imagen** (mejor eso que una foto engañosa).
- Las imágenes se reducen a máx. 400 px (JPEG) y se guardan en `data/cache/images/`. Una búsqueda ya hecha no vuelve a llamar a la API.
- Si se interrumpe, vuelve a correr el mismo comando: el progreso se guarda cada 10 tarjetas.
- Los créditos (autor + página de Pixabay) quedan en la metadata de la nota, no en la tarjeta.

| Opción | Default | Qué hace |
|---|---|---|
| `-i/--input` | `data/output/cards.json` | JSON de entrada |
| `-o/--output` | el mismo de entrada | JSON de salida |
| `--limit` | — | Procesa solo N tarjetas (escribe todas) |
| `--overrides` | `data/overrides.yaml` | Ajustes manuales |

**Ajustes manuales** (`data/overrides.yaml`): la llave es `lema` (todas sus categorías) o `lema/POS` (solo esa tarjeta; gana sobre `lema`). Cada entrada lleva exactamente una opción:

```yaml
bat/NOUN:
  query: bat animal flying        # otra búsqueda en Pixabay
knife:
  image: data/overrides/knife.jpg # imagen propia (ruta relativa al proyecto)
damn:
  skip: true                      # sin imagen
```

Los overrides se aplican en cada corrida, aunque la tarjeta ya tenga imagen.

### Audio (voces neuronales)

Usa [edge-tts](https://github.com/rany2/edge-tts) (voces neuronales de Microsoft Edge; no requiere API key, sí conexión a internet). Por cada tarjeta genera dos `.mp3`: la palabra y la oración de ejemplo en inglés.

```bash
# Prueba con 10 tarjetas (actualiza data/output/cards.json)
uv run anki-series audio --limit 10

# Todas, con otra voz
uv run anki-series audio --voice en-US-AriaNeural

# Ver las voces disponibles
uv run edge-tts --list-voices | grep en-US
```

- La voz por default es `TTS_VOICE` (en `.env`; default `en-US-AndrewNeural`).
- Los audios se guardan en `data/cache/audio/` con un nombre que depende de (voz, texto): el mismo texto con la misma voz nunca se vuelve a generar. Cambiar de voz genera audios nuevos; para regenerar uno, borra su `.mp3`.
- Se generan varios audios en paralelo (`--concurrency`, default 4). Los errores de red o del servicio se reintentan hasta 3 veces con espera creciente.
- Si se interrumpe, vuelve a correr el mismo comando: el progreso se guarda cada 10 tarjetas.

| Opción | Default | Qué hace |
|---|---|---|
| `-i/--input` | `data/output/cards.json` | JSON de entrada |
| `-o/--output` | el mismo de entrada | JSON de salida |
| `--limit` | — | Procesa solo N tarjetas sin audio (escribe todas) |
| `--voice` | `TTS_VOICE` | Voz de edge-tts |
| `--concurrency` | 4 | Audios generados en paralelo |

### Construir el mazo

```bash
# Lee data/output/cards.json (si existe) o words.json → data/output/mi-serie.apkg
uv run anki-series build

# Rutas explícitas
uv run anki-series build -i data/output/words.json -o data/output/mi-serie.apkg
```

Cada tarjeta (inglés → español) tiene:

- **Frente**: solo la palabra y su categoría (+ botón de audio arriba a la derecha, si corriste `audio`).
- **Reverso**: palabra + audio, traducción (vacía si no corriste `translate`), imagen centrada (si corriste `images` y se encontró una) y la oración de la serie con la palabra resaltada, su audio y la traducción en cursiva.
- **Audio**: si corriste `audio`, los `.mp3` de la palabra y de la oración van dentro del `.apkg` (con nombres únicos `anki-series-<hash>.mp3`); si no, los botones no aparecen. No se usa el TTS nativo del sistema (sonaba mal).
- **Metadata oculta**: créditos de la imagen y archivo/timestamp de origen.

Volver a importar un `.apkg` regenerado **actualiza** las notas existentes en lugar de duplicarlas (el GUID depende solo de serie + lema + categoría). Tu progreso de estudio se conserva.

## Desarrollo

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Uso de los datos

Los subtítulos y videos tienen copyright: viven en `data/` (ignorado por git) y no se suben al repositorio. Las imágenes provienen de Pixabay, cuya licencia permite redistribuirlas; el mazo está pensado para uso personal y para compartir con amigos cercanos.
