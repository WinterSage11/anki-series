# anki-series

Herramienta de línea de comandos que toma los subtítulos (`.srt`) en inglés de una serie y genera un mazo de Anki (`.apkg`) con:

- Las 1,000 palabras más usadas (lema + categoría gramatical).
- Una oración real de la serie como ejemplo.
- Traducción al español de México/LATAM considerando el contexto de la oración.
- Una imagen cuando aporta valor (empaquetada dentro del `.apkg`).

> Estado: **Fase 1** (lectura y limpieza de subtítulos). Ver el plan de fases en [CLAUDE.md](CLAUDE.md).

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
```

## Desarrollo

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Uso de los datos

Los subtítulos y videos tienen copyright: viven en `data/` (ignorado por git) y no se suben al repositorio. Las imágenes provienen de Pixabay, cuya licencia permite redistribuirlas; el mazo está pensado para uso personal y para compartir con amigos cercanos.
