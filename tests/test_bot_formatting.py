import pytest

from bot.formatting import split_answer


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
