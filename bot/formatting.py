"""Подготовка обычного текста к отправке в Telegram."""


def split_answer(text: str, limit: int = 4096) -> list[str]:
    """Разбить текст без потерь, консервативно считая длину в UTF-16.

    Emoji вне BMP занимают две единицы. Предпочитаем границы абзацев
    или слов, но умеем делить и длинные строки без пробелов.
    """
    if limit < 2:
        raise ValueError("limit должен быть не меньше 2")
    parts = []
    while text:
        size = 0
        end = 0
        boundary = 0
        for index, char in enumerate(text):
            size += 2 if ord(char) > 0xFFFF else 1
            if size > limit:
                break
            end = index + 1
            if char.isspace():
                boundary = end
        if end < len(text) and boundary:
            end = boundary
        parts.append(text[:end])
        text = text[end:]
    return parts
