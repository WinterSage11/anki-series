"""Tests de expresiones multipalabra. OpenAI es un doble en memoria (``FakeClient``)."""

from pathlib import Path

import pytest
from apkg_factory import make_apkg
from test_translate import FakeClient, completion, connection_error, no_sleep

from anki_series import expressions
from anki_series.expressions import (
    Candidate,
    ExpressionBatch,
    ExpressionCache,
    ExpressionError,
    ExpressionVerdict,
    collect_candidates,
    filter_candidates,
    find_expression,
    merge_ranked,
    select_expressions,
)
from anki_series.models import Card, Sentence, SourceRef
from anki_series.subtitles import load_sentences
from anki_series.vocab import known_expressions_from_words, load_nlp, read_known_words

MODEL = "gpt-test"
FIXTURE = Path(__file__).parent / "fixtures" / "expressions.srt"


def sents(*texts: str) -> list[Sentence]:
    return [
        Sentence(text=t, source=SourceRef(file="x.srt", start=f"00:00:{i:02d},000"))
        for i, t in enumerate(texts)
    ]


def texts(candidates) -> dict[str, str]:
    return {c.text: c.hint for c in candidates}


def run_filter(candidates, tmp_path: Path, client: FakeClient | None = None, **kw):
    client = client or FakeClient()
    cache = ExpressionCache(tmp_path / "expressions.json")
    pairs, new = filter_candidates(
        candidates, cache=cache, model=MODEL, client_factory=lambda: client, **kw
    )
    return pairs, new, client, cache


# --- Candidatos ------------------------------------------------------------------------


def test_separable_phrasal_verbs_share_lemma():
    found = collect_candidates(
        sents("I'll figure it out tomorrow.", "We need to figure out what happened.")
    )
    assert texts(found) == {"figure out": "phrasal"}
    assert found[0].frequency == 2


def test_phrasal_verb_with_bound_preposition():
    found = collect_candidates(sents("I can't put up with this.", "Nobody puts up with him."))
    assert "put up with" in texts(found)


def test_ngrams_need_three_occurrences_and_drop_noise():
    found = texts(collect_candidates(load_sentences(FIXTURE)))
    assert found == {
        "figure out": "phrasal",
        "get rid of": "ngram",  # "get rid" queda absorbido por el más largo
        "make sense": "ngram",
        "no offense": "ngram",
        "sit down": "phrasal",  # literal: lo descarta el filtro del LLM, no spaCy
    }
    twice = texts(collect_candidates(sents("No offense, man.", "No offense to you.")))
    assert "no offense" not in twice


def test_ngrams_with_names_or_only_grammar_are_dropped():
    lines = ["Harvey said no.", "Harvey said yes.", "Harvey said hi."]
    lines += ["I'm going to the store.", "We're going to the bank.", "Going to the party?"]
    found = texts(collect_candidates(sents(*lines)))
    assert not any("harvey" in t for t in found)
    assert "go to the" not in found and "go to" not in found


# --- Filtro con OpenAI (mock) ----------------------------------------------------------


def test_filter_discards_literal_and_normalizes(tmp_path: Path):
    candidates = collect_candidates(load_sentences(FIXTURE))
    pairs, new, client, _ = run_filter(candidates, tmp_path)
    assert new == len(candidates) and len(client.calls) == 1
    kept = {v.expression: v.kind for _c, v in pairs if v.keep}
    assert kept == {
        "figure out": "phrasal",
        "get rid of": "idiom",
        "make sense": "idiom",
        "no offense": "idiom",
    }


def test_filter_sends_batches_and_saves_cache_per_batch(tmp_path: Path):
    candidates = [Candidate(f"word{i} up", "ngram", 3, sents("x")[0]) for i in range(5)]
    saves: list[int] = []
    cache_path = tmp_path / "expressions.json"

    def on_batch(done: int, total: int) -> None:
        saves.append(len(ExpressionCache(cache_path)))

    pairs, new, client, _ = run_filter(candidates, tmp_path, batch_size=2, on_batch=on_batch)
    assert len(client.calls) == 3 and new == 5 and len(pairs) == 5
    assert saves == [2, 4, 5]  # el caché ya está en disco al terminar cada lote


def test_filter_cache_avoids_second_call_and_needs_no_client(tmp_path: Path):
    candidates = collect_candidates(load_sentences(FIXTURE))
    run_filter(candidates, tmp_path)

    def no_client():
        raise AssertionError("no debería crear cliente")

    cache = ExpressionCache(tmp_path / "expressions.json")
    pairs, new = filter_candidates(candidates, cache=cache, model=MODEL, client_factory=no_client)
    assert new == 0 and len(pairs) == len(candidates)


def test_filter_failure_keeps_previous_batches(tmp_path: Path):
    candidates = [Candidate(f"word{i} up", "ngram", 3, sents("x")[0]) for i in range(4)]
    client = FakeClient(script=[None, connection_error(), connection_error()])
    with pytest.raises(ExpressionError):
        run_filter(candidates, tmp_path, client, batch_size=2, max_retries=1, sleep=no_sleep)
    assert len(ExpressionCache(tmp_path / "expressions.json")) == 2


def test_filter_normalizes_verdicts(tmp_path: Path):
    candidates = [Candidate("figures out", "phrasal", 3, sents("x")[0]), Candidate("a b", "ngram")]
    answer = ExpressionBatch(
        items=[
            ExpressionVerdict(id=0, keep=True, expression="  Figure   OUT ", kind="phrasal"),
            ExpressionVerdict(id=1, keep=True, expression="single", kind="idiom"),
        ]
    )
    pairs, *_ = run_filter(candidates, tmp_path, FakeClient(script=[completion(answer)]))
    verdicts = [v for _c, v in pairs]
    assert verdicts[0].expression == "figure out"
    assert not verdicts[1].keep  # una sola palabra no es expresión


# --- Ubicar la expresión ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("sentence", "expression", "words"),
    [
        ("I'll figure it out tomorrow.", "figure out", ["figure", "out"]),
        ("Pick the damn kids up!", "pick up", ["Pick", "up"]),
        ("That makes sense to me.", "make sense", ["makes", "sense"]),
        ("I can't put up with this.", "put up with", ["put", "up", "with"]),
        ("Get out, now. Up!", "get up", None),  # la puntuación corta la búsqueda
    ],
)
def test_find_expression(sentence: str, expression: str, words: list[str] | None):
    doc = load_nlp()(sentence)
    found = find_expression(doc, expression)
    assert (found and [doc[i].text for i in found]) == (words or None)


# --- Selección, cupo y mazo conocido ---------------------------------------------------


def selected(tmp_path: Path, **kw) -> list[Card]:
    pairs, *_ = run_filter(collect_candidates(load_sentences(FIXTURE)), tmp_path)
    return select_expressions(pairs, "X", **kw)


def test_select_builds_cards_with_valid_examples(tmp_path: Path):
    cards = selected(tmp_path, limit=10)
    assert [(c.lemma, c.pos) for c in cards] == [
        ("figure out", "phrasal"),
        ("get rid of", "idiom"),
        ("make sense", "idiom"),
        ("no offense", "idiom"),
    ]
    assert [c.rank for c in cards] == [1, 2, 3, 4]
    assert all(c.frequency == 3 for c in cards)
    assert "figure this out" in cards[0].example_en


def test_select_respects_limit(tmp_path: Path):
    assert len(selected(tmp_path, limit=2)) == 2


def test_select_drops_expression_missing_from_examples(tmp_path: Path):
    cand = Candidate("get rid", "ngram", 3, sents("Get rid, man.")[0])
    verdict = ExpressionVerdict(id=0, keep=True, expression="get rid of", kind="idiom")
    assert select_expressions([(cand, verdict)], "X", limit=5) == []


def test_known_deck_expressions_are_excluded(tmp_path: Path):
    notes = [["1", "to figure (sth) out", ""], ["2", "Makes sense", ""], ["3", "car", ""]]
    words = read_known_words(make_apkg(tmp_path, notes=notes), "Palabra")
    known = known_expressions_from_words(words)
    assert {"figure out", "make sense"} <= known
    assert "car" not in known
    lemmas = {c.lemma for c in selected(tmp_path, limit=10, known=known)}
    assert lemmas == {"get rid of", "no offense"}


def test_merge_ranked_interleaves_by_frequency():
    src = SourceRef(file="x.srt", start="0")

    def card(lemma: str, pos: str, freq: int) -> Card:
        return Card(series="X", lemma=lemma, pos=pos, rank=1, frequency=freq,
                    example_en="x", source=src)  # fmt: skip

    merged = merge_ranked([card("car", "NOUN", 5), card("bail", "VERB", 1)],
                          [card("figure out", "phrasal", 3)])  # fmt: skip
    assert [(c.lemma, c.rank) for c in merged] == [("car", 1), ("figure out", 2), ("bail", 3)]


# --- Presupuesto de candidatos y variantes con preposición -----------------------------


def test_phrasal_verbs_have_their_own_candidate_budget(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(expressions, "MAX_CANDIDATES", 3)
    monkeypatch.setattr(expressions, "MAX_PHRASAL_CANDIDATES", 2)
    lines = ["I'll figure it out.", "We figure it out."]  # phrasal ×2
    lines += [f"{w} the {w} car is great." for w in ("red", "blue", "old", "new")] * 3
    found = collect_candidates(sents(*lines))
    assert len(found) == 3
    assert found[0].text == "figure out"  # entra aunque los n-gramas son más frecuentes
    assert all(c.hint == "ngram" for c in found[1:])


def verdict_pair(expression: str, kind: str, freq: int, sentence: str):
    cand = Candidate(expression, "phrasal", freq, sents(sentence)[0])
    return cand, ExpressionVerdict(id=0, keep=True, expression=expression, kind=kind)


@pytest.mark.parametrize(
    ("freqs", "kept"),
    [((13, 7), "hook up"), ((7, 7), "hook up"), ((3, 9), "hook up with")],
)
def test_preposition_variants_keep_only_one(freqs: tuple[int, int], kept: str):
    pairs = [
        verdict_pair("hook up", "phrasal", freqs[0], "They hooked up last night."),
        verdict_pair("hook up with", "phrasal", freqs[1], "She hooked up with Barney."),
        verdict_pair("break up with", "phrasal", 5, "I broke up with her."),
    ]
    lemmas = {c.lemma for c in select_expressions(pairs, "X", limit=10)}
    assert lemmas == {kept, "break up with"}  # sin "break up" no hay duplicado
