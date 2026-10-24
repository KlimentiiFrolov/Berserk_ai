"""Общие хелперы очистки значений (вынесены из старого google_sheets_normalizer)."""
from __future__ import annotations

import hashlib
import re
from typing import Any


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"_x000[0-9A-Fa-f]_", " ", text)   # управляющие символы из Excel
    text = text.replace(" ", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text or None


def parse_int(value: Any) -> int | None:
    """'12' / '12.0' -> 12, всё остальное -> None."""
    text = clean_text(value)
    if text is None:
        return None
    if re.fullmatch(r"\d+(\.0+)?", text):
        return int(float(text))
    return None


def parse_checkbox(value: Any) -> bool | None:
    """Чекбокс Google Sheets в CSV-экспорте: TRUE / FALSE / пусто."""
    text = clean_text(value)
    if text is None:
        return None
    low = text.casefold()
    if low in {"true", "1", "1.0", "yes", "да", "✓", "✔"}:
        return True
    if low in {"false", "0", "0.0", "no", "нет"}:
        return False
    return None


def short_hash(*parts: Any, length: int = 12) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def slugify(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[^0-9a-zа-яё]+", "-", value)
    return value.strip("-")
