"""Подготовка обычного текста к отправке в Telegram."""

from rag.schemas import Source


def format_sources(sources: list[Source]) -> str:
    """Источники из RAG без генерации ссылок или интерпретации названий."""
    if not sources:
        return ""
    seen = set()
    lines = ["Источники:"]
    for source in sources:
        key = (source.collection, source.source_type, source.title, source.url)
        if key in seen:
            continue
        seen.add(key)
        label = "официальный источник" if source.source_type == "official" else "сообщество"
        lines.append(f"• {source.title} ({label})")
        if source.url:
            lines.append(source.url)
    return "\n\n" + "\n".join(lines)


def split_answer(text: str, limit: int = 4096) -> list[str]:
    """Разбить текст без потерь с лимитом в кодовых единицах UTF-16.

    Символы вне BMP занимают две кодовые единицы. Предпочитаем границы абзацев
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
