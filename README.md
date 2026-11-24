# anki-series

Herramienta de línea de comandos que toma los subtítulos (`.srt`) en inglés de una serie y genera un mazo de Anki (`.apkg`) con:

- Las 1,000 palabras más usadas (lema + categoría gramatical).
- Una oración real de la serie como ejemplo.
- Traducción al español de México/LATAM considerando el contexto de la oración.
- Una imagen cuando aporta valor (empaquetada dentro del `.apkg`).

> Estado: **Fase 4** (vocabulario → `words.json` → traducción con OpenAI → `cards.json` → mazo `.apkg`; sin imágenes todavía). Ver el plan de fases en [CLAUDE.md](CLAUDE.md).

## Requisitos

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 (uv lo instala si no lo tienes)

## Instalación

```bash
uv sync
cp .env.example .env   # y llena las API keys cuando lleguemos a las fases 4 y 5
```

## Uso (previsto)

```bash
# Un archivo o una carpeta con todos los .srt de la temporada/serie
uv run anki-series all data/raw/mi-serie/ --series "Mi Serie" \
    --known-deck data/known/top1000.apkg
# → data/output/mi-serie.apkg
```

Hoy solo existe:

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
| `-o/--output` | `data/output/words.json` | Ruta del JSON de salida |
| `--limit` | 10 | Filas de muestra en pantalla |

Para no repetir `--known-field`, ponlo en `.env` (p. ej. `KNOWN_DECK_FIELD=Palabra`).

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

### Construir el mazo

```bash
# Lee data/output/cards.json (si existe) o words.json → data/output/mi-serie.apkg
uv run anki-series build

# Rutas explícitas
uv run anki-series build -i data/output/words.json -o data/output/mi-serie.apkg
```

Cada tarjeta (inglés → español) tiene:

- **Frente**: solo la palabra y su categoría (+ botón de audio arriba a la derecha, cuando exista).
- **Reverso**: palabra + audio, traducción (vacía si no corriste `translate`), imagen centrada (si hay) y la oración de la serie con la palabra resaltada, su audio y la traducción en cursiva.
- **Audio**: los campos `AudioWord` y `AudioExample` ya existen pero van vacíos; se llenarán con `.mp3` de voces neuronales más adelante (el TTS nativo del sistema sonaba mal).
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
