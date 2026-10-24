"""Восстановление номеров пунктов из автонумерации Word.

В .docx номера вида «105.2.3» не хранятся текстом: у абзаца есть только (numId, ilvl), а шаблон
уровня лежит в numbering.xml (lvlText, например «105.%1.»). Здесь повторяется то, что делает Word:
ведём счётчики по уровням, при шаге на уровне L сбрасываем более глубокие, подставляем значения в lvlText.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..loaders.rules_docx import NumLevel

_ROMAN = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
          (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]


def _roman(n: int) -> str:
    out = ""
    for value, sym in _ROMAN:
        while n >= value:
            out += sym
            n -= value
    return out


def _letters(n: int) -> str:
    out = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("a") + rem) + out
    return out


def format_counter(fmt: str, n: int) -> str:
    if fmt == "lowerLetter":
        return _letters(n)
    if fmt == "upperLetter":
        return _letters(n).upper()
    if fmt == "lowerRoman":
        return _roman(n)
    if fmt == "upperRoman":
        return _roman(n).upper()
    if fmt == "decimalZero":
        return f"{n:02d}"
    return str(n)


@dataclass(frozen=True)
class Label:
    text: str            # как в Word, без хвостовой точки: "105.2.3", "a", "i"
    is_full: bool        # шаблон уровня уже включает номера верхних уровней
    is_bullet: bool = False


class NumberingState:
    def __init__(self, numbering: dict[int, dict[int, NumLevel]]):
        self.defs = numbering
        self.counters: dict[int, dict[int, int]] = {}

    def advance(self, num_id: int, ilvl: int) -> Label | None:
        levels = self.defs.get(num_id)
        if not levels or ilvl not in levels:
            return None
        lvl = levels[ilvl]
        counters = self.counters.setdefault(num_id, {})
        counters[ilvl] = counters.get(ilvl, lvl.start - 1) + 1
        for deeper in [k for k in counters if k > ilvl]:
            del counters[deeper]
        if lvl.fmt in ("bullet", "none"):
            return Label("", False, is_bullet=True)

        def sub(match: re.Match) -> str:
            level = int(match.group(1)) - 1
            spec = levels.get(level)
            if spec is None:
                return ""
            return format_counter(spec.fmt, counters.get(level, spec.start))

        rendered = re.sub(r"%(\d)", sub, lvl.text).strip().rstrip(".)").strip()
        refs = {int(m) - 1 for m in re.findall(r"%(\d)", lvl.text)}
        is_full = any(r < ilvl for r in refs)
        return Label(rendered, is_full)
