import pytest
from pydantic import ValidationError

from anki_series.models import Card, ImageInfo, ImageStatus, SourceRef


def make_card(**overrides) -> Card:
    data = {
        "series": "The Night Shift",
        "lemma": "bail",
        "pos": "VERB",
        "rank": 1,
        "frequency": 3,
        "example_en": "We need to bail before the cops show up.",
        "source": SourceRef(file="S01E01.srt", start="00:00:07,200"),
    }
    data.update(overrides)
    return Card(**data)


def test_new_card_has_pending_image_and_no_translation():
    card = make_card()
    assert card.image.status is ImageStatus.PENDING
    assert card.translation is None
    assert card.example_es is None


def test_card_roundtrip_without_image():
    card = make_card()
    restored = Card.model_validate_json(card.model_dump_json())
    assert restored == card


def test_card_roundtrip_with_image():
    card = make_card(
        translation="largarse",
        example_es="Tenemos que largarnos antes de que llegue la policía.",
        imageable=True,
        image=ImageInfo(
            status=ImageStatus.FOUND,
            query="person running away",
            path="data/cache/images/bail_verb.jpg",
            source="pixabay",
            credit="Pixabay user 123",
        ),
    )
    restored = Card.model_validate_json(card.model_dump_json())
    assert restored == card
    assert restored.image.status is ImageStatus.FOUND


def test_key_combines_series_lemma_and_pos():
    assert make_card().key == "The Night Shift|bail|VERB"


def test_rank_must_be_positive():
    with pytest.raises(ValidationError):
        make_card(rank=0)
