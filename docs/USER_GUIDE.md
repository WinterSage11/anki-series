# anki-series — User Guide

Step-by-step manual for turning a show's English subtitles into an Anki deck. For the project overview and design rationale, see the [README](../README.md).

- [1. Requirements](#1-requirements)
- [2. Configuration (`.env`)](#2-configuration-env)
- [3. Subtitle folder layout](#3-subtitle-folder-layout)
- [4. One command: `all`](#4-one-command-all)
- [5. Stage by stage](#5-stage-by-stage)
- [6. Flag reference](#6-flag-reference)
- [7. Caches and re-runs](#7-caches-and-re-runs)
- [8. Importing and updating the deck in Anki](#8-importing-and-updating-the-deck-in-anki)
- [9. Troubleshooting](#9-troubleshooting)

## 1. Requirements

| Requirement | Notes |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Manages Python, the virtual environment and dependencies. Never install with `pip` globally. |
| Python 3.12+ | `uv` installs it automatically (`.python-version`). |
| spaCy model `en_core_web_sm` | Declared as a dependency in `pyproject.toml` (pinned wheel), so `uv sync` installs it. No `spacy download` needed. |
| OpenAI API key | For `translate` and `--expressions`. |
| Pixabay API key | For `images` ([get one free](https://pixabay.com/api/docs/)). |
| Internet access | For `audio` (edge-tts, no key required). |
| Anki desktop | To import the `.apkg`. |

```bash
uv sync
uv run anki-series --help
```

Want to try it with no keys at all? Run the offline demo:

```bash
uv run anki-series extract examples/sample.srt --series "Demo" --top 30 -o data/demo/words.json
uv run anki-series build -i data/demo/words.json -o data/demo/demo.apkg
```

## 2. Configuration (`.env`)

```bash
cp .env.example .env
```

| Variable | Default | Used by |
|---|---|---|
| `OPENAI_API_KEY` | — | `translate`, `extract --expressions` |
| `OPENAI_MODEL` | `gpt-4o` | `translate`, expression filter. `gpt-4.1-mini` also works; `gpt-4o-mini` handles phrasal verbs poorly. |
| `PIXABAY_API_KEY` | — | `images` |
| `TTS_VOICE` | `en-US-AndrewNeural` | `audio` (list voices: `uv run edge-tts --list-voices`) |
| `KNOWN_DECK_FIELD` | `0` | `--known-deck`: field index or name holding the English word |
| `DATA_DIR` | `data` | Root for `output/`, `cache/` and `overrides.yaml` |

`.env` is git-ignored; never commit it.

## 3. Subtitle folder layout

Input is a single `.srt` or a folder of `.srt` files (a season or a whole series). Keep real subtitles under `data/` — it is git-ignored because subtitles are copyrighted.

```
data/
├── raw/
│   └── my-show/            # input: one folder per show/season
│       ├── S04E01.srt
│       └── S04E02.srt
├── known/
│   └── my-deck.apkg        # optional: a deck with words you already know
├── overrides.yaml          # optional: manual image fixes
├── output/                 # generated: words.json, cards.json, <series>.apkg
└── cache/                  # generated: translations, expressions, images, audio
```

Encoding is detected automatically. Subtitle credits (`== synced & corrected by … ==`, `Subtitles by …`, OpenSubtitles/Addic7ed, URLs) are dropped, even when split across two lines.

## 4. One command: `all`

```bash
# Cheap trial: only 20 cards through translate/images/audio/build
uv run anki-series all data/raw/my-show/ --series "My Show" --exclude-common 500 --limit 20

# Full run
uv run anki-series all data/raw/my-show/ --series "My Show" \
    --top 350 --expressions 50 --exclude-common 500 \
    --known-deck data/known/my-deck.apkg --known-field Word
# → data/output/my-show.apkg
```

`all` runs `extract → translate → images → audio → build` using the default paths in `data/output/`. It stops at the first failing stage; fix the cause (network, key, quota) and run the **same command** again — finished work is read from cache and not paid for twice.

## 5. Stage by stage

Useful to inspect the intermediate JSON or redo a single stage.

| Stage | Command | Reads → writes |
|---|---|---|
| 1. Vocabulary | `extract <path> --series NAME` | `.srt` → `data/output/words.json` |
| 2. Translation | `translate` | `words.json` → `cards.json` |
| 3. Images | `images` | `cards.json` → `cards.json` (in place) |
| 4. Audio | `audio` | `cards.json` → `cards.json` (in place) |
| 5. Deck | `build` | `cards.json` (or `words.json`) → `<series>.apkg` |

```bash
# Preview cleaned sentences without writing anything
uv run anki-series extract data/raw/my-show/ --series "My Show" --dry-run --limit 20

# 1. Vocabulary (offline unless --expressions > 0)
uv run anki-series extract data/raw/my-show/ --series "My Show" --top 350 --expressions 50

# 2. Translate (try 10 first)
uv run anki-series translate --limit 10
uv run anki-series translate

# 3. Images, 4. Audio
uv run anki-series images
uv run anki-series audio --voice en-US-AriaNeural

# 5. Build the deck
uv run anki-series build
```

You can `build` at any point: missing translations, images or audio simply leave those fields empty.

### What gets filtered in `extract`

- Proper nouns (`PROPN` / NER `PERSON`); if more than half of a lemma's occurrences are tagged as names, it is dropped for every POS (e.g. a character called *Harvey*). Side effect: `mom`/`dad` used as a vocative may disappear.
- Interjections (`yeah`, `oh`, `okay`, `hey`…), vocal fillers (`hmm`, `uh`, `um`, `mhm`…), stopwords, numbers and punctuation. Slang and swear words are kept.
- `gonna` / `gotta` / `wanna` count as `go` / `get` / `want` (verb).
- One card per lemma: the most frequent POS wins and its frequency is the sum over all POS.
- Example sentence: 4–20 words, closest to 10; ties → first occurrence.

### Multiword expressions (`--expressions N`)

spaCy proposes phrasal verbs (including separable ones: "figure **it** out") and frequent 2–4-word n-grams; OpenAI then keeps only expressions with their own meaning (`get rid of`, `make sense`) and labels them `phrasal` or `idiom`. They take up to N of the `--top` slots and are mixed with words by frequency. They are checked against `--known-deck` but not against `--exclude-common`.

### Image overrides (`data/overrides.yaml`)

Key = `lemma` (all POS) or `lemma/POS` (wins over `lemma`). Exactly one option per entry; overrides apply on every run.

```yaml
bat/NOUN:
  query: bat animal flying        # different Pixabay search
knife:
  image: data/overrides/knife.jpg # your own image (path relative to the project)
damn:
  skip: true                      # no image
```

Only words the model marks as illustratable are searched. A Pixabay hit is used only if its tags share a word with the query — no image is better than a misleading one. Images are resized to max 400 px JPEG.

## 6. Flag reference

### `all`

| Flag | Default | What it does |
|---|---|---|
| `INPUT_PATH` | — (required) | `.srt` file or folder of `.srt` |
| `--series` | — (required) | Series name: defines the deck (`anki-series::<series>`) and note GUIDs |
| `--top` | 1000 | Total cards to keep (words + expressions) |
| `--known-deck` | — | `.apkg` with words you already know; their lemmas are excluded |
| `--known-field` | `KNOWN_DECK_FIELD` or `0` | Field of the known deck with the English word (index or name) |
| `--exclude-common` | 0 | Excludes the N most frequent English words (`wordfreq`); e.g. 500 drops `get`, `go`, `know`, `time`. 0 = off |
| `--expressions` | 0 | Reserves up to N of `--top` for phrasal verbs and idioms (uses OpenAI). 0 = words only |
| `--limit` | — | Only passed to `translate`: `cards.json` keeps the first N cards, and later stages process those |
| `--model` | `OPENAI_MODEL` | OpenAI model for translation and the expression filter |
| `--voice` | `TTS_VOICE` | edge-tts voice |
| `--overrides` | `data/overrides.yaml` | Image overrides file |
| `--concurrency` | 4 | Audio files generated in parallel (1–16) |

### Per-command flags

| Command | Flag | Default | What it does |
|---|---|---|---|
| `extract` | `--top`, `--known-deck`, `--known-field`, `--exclude-common`, `--expressions`, `--model` | as in `all` | Same meaning as above |
| `extract` | `--dry-run` | off | Only prints cleaned sentences; writes nothing |
| `extract` | `--limit` | 10 | Rows shown on screen (not a processing limit here) |
| `extract` | `-o/--output` | `data/output/words.json` | Output JSON |
| `translate` | `-i/--input` / `-o/--output` | `words.json` / `cards.json` | Input / output JSON |
| `translate` | `--limit` | — | Only the first N cards (writes only those) |
| `translate` | `--batch-size` | 20 | Cards per API call (1–100) |
| `translate` | `--model` | `OPENAI_MODEL` | Model for this run |
| `images` | `-i/--input` / `-o/--output` | `cards.json` / same as input | Input / output JSON |
| `images` | `--limit` | — | Process only N pending cards (still writes all) |
| `images` | `--overrides` | `data/overrides.yaml` | Manual fixes |
| `audio` | `-i/--input` / `-o/--output` | `cards.json` / same as input | Input / output JSON |
| `audio` | `--limit` | — | Process only N cards without audio (still writes all) |
| `audio` | `--voice`, `--concurrency` | `TTS_VOICE`, 4 | Voice and parallelism |
| `build` | `-i/--input` | `cards.json`, else `words.json` | Cards JSON |
| `build` | `-o/--output` | `data/output/<series>.apkg` | Deck path |

### Known deck (`--known-deck`)

Reads `collection.anki21b` (zstd) › `.anki21` › `.anki2` from the `.apkg`, takes the chosen field, strips HTML and `[sound:]` tags and lemmatizes it. Single words exclude lemmas; multiword entries (`give up`, `to figure (sth) out`) exclude expressions. If the field has no words, the error lists the available fields. Set `KNOWN_DECK_FIELD=Word` in `.env` to avoid repeating `--known-field`.

## 7. Caches and re-runs

Everything paid or slow is cached under `data/cache/` and written atomically (temp file + rename), so an interrupted run never corrupts it.

| Cache | Key | To force a refresh |
|---|---|---|
| `translations.json` | prompt version + model + lemma + POS + sentence | change model/sentence, or delete the file |
| `expressions.json` | filter version + model + candidate text | delete the file |
| `images/` + `index.json` | sha1 of the query (also stores "not found") | delete the entry in `index.json` |
| `audio/*.mp3` | sha1 of voice + text | delete the `.mp3` |

- Translations and expressions are saved after every batch; `images` and `audio` checkpoint `cards.json` every 10 cards and on error / `Ctrl+C`.
- Transient errors (network, timeout, 429, 5xx) are retried 3 times with 2/4/8 s backoff; other errors stop immediately.
- Re-running with a warm cache makes **zero** API calls.
- The translation cache does not include the series name, so the same word + sentence is reused across shows.

## 8. Importing and updating the deck in Anki

1. In Anki desktop: **File → Import…** and choose `data/output/<series>.apkg`.
2. The deck appears as `anki-series::<series>`. Each note is tagged with its POS (`NOUN`, `VERB`, `phrasal`, `idiom`…), useful for filtered decks.
3. Card layout: **front** = word + POS (+ audio button); **back** = word, translation, image, the example from the show with the word in bold, its audio and the Spanish translation in italics. Image credits and source timestamp are stored in hidden fields.

**Updating:** regenerate the deck (e.g. after adding images or audio) and import it again. Notes are matched by GUID (`series + lemma + POS`), so they are **updated in place** and your review history is kept. Different series never mix because each one gets its own deck ID.

Note: Anki does not replace existing media files on re-import; that is why media names include a content hash (`anki-series-<hash>.jpg/.mp3`) and change whenever the image, voice or text changes.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Falta OPENAI_API_KEY` / `Falta PIXABAY_API_KEY` | Add the key to `.env` |
| `--known-field` error listing fields | Pick the field name/index that holds the English word |
| A word got a bad image | Add an override (`query`, `image` or `skip`) and re-run `images` + `build` |
| Translation looks off after editing the prompt | Bump `PROMPT_VERSION` (words) or `EXPRESSION_PROMPT_VERSION` (expressions) in `translate.py` |
| Run stopped halfway | Re-run the same command; only the missing work is done |
