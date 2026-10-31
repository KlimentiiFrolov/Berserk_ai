import pytest

from bot.formatting import format_sources, split_answer
from rag.schemas import Source


def test_sources_labels_optional_urls_and_duplicates():
    official = Source("rules", "official", "Правила: 3.2", "https://example.org/rules")
    community = Source("community", "community", "Разбор <атаки> & защиты")
    assert format_sources([official, community, official]) == (
        "\n\nИсточники:\n"
        "• Правила: 3.2 (официальный источник)\nhttps://example.org/rules\n"
        "• Разбор <атаки> & защиты (сообщество)"
    )


def test_no_sources_has_no_heading():
    assert format_sources([]) == ""


def test_sources_with_same_title_but_distinct_urls_are_preserved():
    sources = [
        Source("cards", "official", "Карта: Пример", url)
        for url in ("https://example.org/1", "https://example.org/2")
    ]
    assert format_sources(sources).count("• Карта: Пример") == 2


@pytest.mark.parametrize(
    "text", ["", "Короткий ответ", "а" * 10000, "Правило атаки.\n\n" * 1000, "🃏" * 5000]
)
def test_split_preserves_text_and_telegram_limit(text):
    parts = split_answer(text)
    assert "".join(parts) == text
    assert all(0 < len(part.encode("utf-16-le")) // 2 <= 4096 for part in parts)


def test_split_prefers_word_boundary():
    assert split_answer("Один два три", limit=9) == ["Один два ", "три"]


def test_invalid_limit():
    with pytest.raises(ValueError):
        split_answer("текст", limit=1)
