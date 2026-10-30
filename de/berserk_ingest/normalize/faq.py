"""Нормализация строк FAQ-вкладки в записи «карта / вопрос / ответ / статус».

Особенности таблицы (проверено на вкладке «4. Ложные боги»):
- «Номер» и «Название» заполнены только в первой строке блока карты, дальше пусто;
- «Вопрос» может быть пустым: тогда строка — просто уточнение по карте (kind=ruling);
- есть служебная строка-преамбула «Это FAQ по картам ...» — не контент;
- «Ответ утвержден» — чекбокс (TRUE/FALSE/пусто).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .text import clean_text, parse_checkbox, parse_int

COLUMN_ALIASES = {
    "number": ("номер", "№"),
    "title": ("название", "карта"),
    "question": ("вопрос",),
    "answer": ("ответ",),
    "approved": ("ответ утвержден", "ответ утверждён", "утвержден", "утверждён"),
}
FALLBACK_COLUMNS = {"number": 0, "title": 1, "question": 2, "answer": 3, "approved": 4}


@dataclass
class FaqRecord:
    set_name: str
    card_number: int | None
    card_name: str
    question: str | None
    answer: str
    approval: Literal["approved", "not_approved", "unknown"]
    kind: Literal["qna", "ruling"]
    source_row: int           # номер строки в CSV (с 1) — для отладки


@dataclass
class FaqReport:
    rows: int = 0
    qna: int = 0
    rulings: int = 0
    unanswered: int = 0            # есть вопрос, нет ответа — пропущены
    cards_without_content: int = 0  # карты без единой записи
    approved: int = 0
    not_approved: int = 0
    unknown: int = 0
    header_row: int | None = None
    warnings: list[str] = field(default_factory=list)


def _find_columns(rows: list[list[str]], report: FaqReport) -> tuple[dict[str, int], int]:
    """Ищет строку заголовков в первых 10 строках; при неудаче — позиции по умолчанию."""
    for idx, row in enumerate(rows[:10]):
        cells = [(clean_text(c) or "").casefold() for c in row]
        if "вопрос" in cells and any(c == "ответ" for c in cells):
            mapping: dict[str, int] = {}
            for key, aliases in COLUMN_ALIASES.items():
                for col, cell in enumerate(cells):
                    if cell in aliases and key not in mapping:
                        mapping[key] = col
                        break
            # «ответ» не должен совпасть с «ответ утвержден»
            for col, cell in enumerate(cells):
                if cell == "ответ":
                    mapping["answer"] = col
            if {"title", "question", "answer"} <= mapping.keys():
                mapping.setdefault("number", FALLBACK_COLUMNS["number"])
                mapping.setdefault("approved", FALLBACK_COLUMNS["approved"])
                return mapping, idx
    report.warnings.append("не найдена строка заголовков, использованы позиции колонок A-E")
    return dict(FALLBACK_COLUMNS), 0


def _approval(raw: str | None) -> Literal["approved", "not_approved", "unknown"]:
    parsed = parse_checkbox(raw)
    if parsed is True:
        return "approved"
    if parsed is False:
        return "not_approved"
    return "unknown"


def normalize_faq(rows: list[list[str]], set_name: str) -> tuple[list[FaqRecord], FaqReport]:
    report = FaqReport(rows=len(rows))
    cols, header_idx = _find_columns(rows, report)
    report.header_row = header_idx + 1

    def cell(row: list[str], key: str) -> str | None:
        col = cols.get(key)
        return clean_text(row[col]) if col is not None and col < len(row) else None

    records: list[FaqRecord] = []
    cards_seen: dict[str, int] = {}   # название -> число записей
    cur_number: int | None = None
    cur_name: str | None = None

    for offset, row in enumerate(rows[header_idx + 1:], start=header_idx + 2):
        title = cell(row, "title")
        if title and title.startswith("Это FAQ"):
            continue                                   # преамбула вкладки

        number = parse_int(cell(row, "number"))
        if number is not None or title:
            # начался новый блок карты
            if title:
                cur_name = title
                cards_seen.setdefault(title, 0)
            if number is not None:
                cur_number = number

        question, answer = cell(row, "question"), cell(row, "answer")
        if not question and not answer:
            continue
        if cur_name is None:
            report.warnings.append(f"строка {offset}: запись без карты, пропущена")
            continue
        if question and not answer:
            report.unanswered += 1
            continue

        approval = _approval(cell(row, "approved"))
        kind: Literal["qna", "ruling"] = "qna" if question else "ruling"
        records.append(FaqRecord(set_name, cur_number, cur_name, question, answer or "", approval, kind, offset))
        cards_seen[cur_name] = cards_seen.get(cur_name, 0) + 1

        if kind == "qna":
            report.qna += 1
        else:
            report.rulings += 1
        setattr(report, approval, getattr(report, approval) + 1)

    report.cards_without_content = sum(1 for n in cards_seen.values() if n == 0)
    return records, report
