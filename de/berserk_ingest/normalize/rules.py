"""Структура правил: абзацы docx -> пункты с номерами, иерархией и контекстом.

Правила разбора:
- Heading 1 «Часть N...» задаёт часть; всё до первой части (разработчики, оглавление) игнорируется;
- Heading 2 «105. Базовые свойства.» задаёт раздел; Heading 3 — подраздел (попадает в путь);
- нумерованный абзац = пункт; его номер собирается из numbering.xml (см. numbering.py);
- маркированные абзацы и ненумерованные продолжения склеиваются с предыдущим пунктом;
- исключение: раздел, где ещё не было ни одного нумерованного пункта (Часть 7, терминология) —
  там каждый маркер верхнего уровня это отдельная запись (термин);
- таблица = отдельный пункт.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from ..loaders.rules_docx import RawRules
from .numbering import NumberingState
from .text import slugify

_PART_RE = re.compile(r"^Часть\s+(\d+)\b")
_SECTION_RE = re.compile(r"^(\d{3})\.\s*(.+?)\.?\s*$")
_REF_RE = re.compile(r"(?<![\d.])(\d{3}(?:\.\w+)+)(?![\w])")


@dataclass
class RuleItem:
    point_id: str
    level: int
    parent_id: str | None
    part: str
    part_no: str
    section: str
    section_id: str
    subsection: str | None
    kind: Literal["point", "term", "table", "note"]
    lines: list[str]
    plain_tail: bool = False       # последняя строка — ненумерованное продолжение (кандидат в подпись таблицы)
    path_ids: list[str] = field(default_factory=list)   # цепочка предков + сам пункт

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class RulesReport:
    paragraphs: int = 0
    items: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    id_collisions: int = 0
    section_mismatches: list[str] = field(default_factory=list)
    refs_total: int = 0
    refs_resolved: int = 0
    refs_unresolved: list[str] = field(default_factory=list)
    symbols_total: int = 0
    symbols_unnamed: int = 0
    warnings: list[str] = field(default_factory=list)


def _assign_point_id(rendered: str, is_full: bool, parent: str | None, section_id: str) -> str:
    """Id пункта из подписи Word.

    Подпись бывает полной («105.2.3»), частичной («a», «2.2.1» — шаблон без префикса раздела)
    или принадлежащей другому разделу («106.1» внутри 105: такой id сохраняется как есть,
    а расхождение попадает в отчёт section_mismatches).
    """
    if is_full and rendered.startswith(section_id + "."):
        return rendered
    if parent and rendered.startswith(parent + "."):
        return rendered
    if parent:
        return f"{parent}.{rendered.split('.')[-1]}"
    if rendered.startswith(section_id + ".") or re.match(r"^\d{3}\.", rendered):
        return rendered
    return f"{section_id}.{rendered}"


def normalize_rules(raw: RawRules) -> tuple[list[RuleItem], RulesReport]:
    report = RulesReport(paragraphs=len(raw.paragraphs))
    numbering = NumberingState(raw.numbering)

    items: list[RuleItem] = []
    by_id: dict[str, RuleItem] = {}
    used: Counter[str] = Counter()

    part = part_no = ""
    section = section_id = ""
    subsection: str | None = None
    started = False
    stack: list[tuple[int, str]] = []        # (уровень, point_id) предков в текущем разделе
    last_decimal: str | None = None          # последний пункт с десятичной нумерацией в разделе
    cur: RuleItem | None = None
    numbered_in_section = False
    note_n = table_n = 0

    def new_item(pid: str, level: int, parent: str | None, kind: str, first_line: str, plain: bool = False) -> RuleItem:
        nonlocal cur
        if used[pid]:                       # коллизия номера: не теряем пункт, помечаем суффиксом
            report.id_collisions += 1
            used[pid] += 1
            pid = f"{pid}~{used[pid]}"
        else:
            used[pid] = 1
        chain = (by_id[parent].path_ids if parent in by_id else []) + [pid]
        item = RuleItem(pid, level, parent, part, part_no, section, section_id, subsection,
                        kind, [first_line] if first_line else [], plain, chain)  # type: ignore[arg-type]
        items.append(item)
        by_id[pid] = item
        cur = item
        return item

    for p in raw.paragraphs:
        text = p.text.strip()

        # ---- заголовки ----
        if p.kind == "p" and p.style.startswith("Heading") and not text:
            continue                        # пустой заголовок: НЕ сбрасывать текущий раздел
        if p.kind == "p" and p.style == "Heading 1":
            m = _PART_RE.match(text)
            if m:
                started = True
                part, part_no = text.rstrip("."), m.group(1)
                section, section_id, subsection = part, part_no, None
                stack, cur, numbered_in_section, note_n, last_decimal = [], None, False, 0, None
            continue
        if not started:
            continue
        if p.kind == "p" and p.style == "Heading 2":
            m = _SECTION_RE.match(text)
            if m:
                section, section_id = text.rstrip("."), m.group(1)
            else:
                section, section_id = text.rstrip("."), f"{part_no}.{slugify(text)[:30]}"
                report.warnings.append(f"заголовок раздела без номера (id из названия): {text!r}")
            subsection = None
            stack, cur, numbered_in_section, note_n, last_decimal = [], None, False, 0, None
            continue
        if p.kind == "p" and p.style == "Heading 3":
            subsection = text.rstrip(".")
            stack, cur, numbered_in_section, last_decimal = [], None, False, None
            continue

        # ---- таблица ----
        if p.kind == "table":
            if not text:
                continue
            caption = None
            if cur and cur.plain_tail and len(cur.lines) > 1 and len(cur.lines[-1]) <= 80:
                caption = cur.lines.pop().rstrip(".")
            parent = stack[-1][1] if stack else None
            table_n += 1
            pid = f"{parent or section_id}.t{table_n}"
            item = new_item(pid, (stack[-1][0] + 1) if stack else 0, parent, "table",
                            (caption if caption.startswith("Таблица") else f"Таблица: {caption}") if caption else "Таблица")
            item.lines.append(text)
            continue

        if p.kind != "p" or not text:
            continue

        # ---- нумерованный / маркированный абзац ----
        if p.num_id is not None:
            label = numbering.advance(p.num_id, p.ilvl or 0)
            if label is not None and not label.is_bullet:
                lvl = p.ilvl or 0
                fmt = raw.numbering[p.num_id][lvl].fmt
                if fmt != "decimal" and lvl == 0 and last_decimal in by_id:
                    # самостоятельный список a), b), c) после пункта: в тексте правил на него
                    # ссылаются как на подпункт («п. 306.3.b»), поэтому родитель — последний пункт
                    parent = last_decimal
                    pid = f"{parent}.{label.text.split('.')[-1]}"
                    item = new_item(pid, by_id[parent].level + 1, parent, "point", text)
                    numbered_in_section = True
                    continue
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                parent = stack[-1][1] if stack else None
                pid = _assign_point_id(label.text, label.is_full, parent, section_id)
                first = pid.split(".")[0]
                if section_id.isdigit() and len(section_id) == 3 and first != section_id:
                    report.section_mismatches.append(f"{pid} в разделе {section_id}")
                item = new_item(pid, lvl, parent, "point", text)
                stack.append((lvl, item.point_id))
                if fmt == "decimal":
                    last_decimal = item.point_id
                numbered_in_section = True
                continue
            # маркер
            if label is not None and label.is_bullet:
                if not numbered_in_section and (p.ilvl or 0) == 0:
                    base = slugify(re.split(r"\.\s|:\s", text, maxsplit=1)[0])[:48] or "term"
                    pid = f"{section_id}.{base}"
                    new_item(pid, 0, None, "term", text)
                    stack = []
                elif cur is not None:
                    cur.lines.append(f"– {text}")
                    cur.plain_tail = False
                else:
                    note_n += 1
                    new_item(f"{section_id}.p{note_n}", 0, None, "note", f"– {text}")
                continue

        # ---- ненумерованный абзац ----
        if cur is not None:
            cur.lines.append(text)
            cur.plain_tail = True
        else:
            note_n += 1
            new_item(f"{section_id}.p{note_n}", 0, None, "note", text, plain=True)

    # ---- отчёт и проверка перекрёстных ссылок ----
    report.items = len(items)
    report.by_kind = dict(Counter(i.kind for i in items))
    known = set(by_id)
    sections = {i.section_id for i in items}
    for item in items:
        for ref in _REF_RE.findall(item.text):
            report.refs_total += 1
            if ref in known or ref in sections:
                report.refs_resolved += 1
            else:
                report.refs_unresolved.append(f"{ref} (в {item.point_id})")
    report.symbols_total = len(raw.symbols)
    report.symbols_unnamed = sum(1 for s in raw.symbols.values() if not s.name)
    report.warnings.extend(raw.warnings)
    return items, report
