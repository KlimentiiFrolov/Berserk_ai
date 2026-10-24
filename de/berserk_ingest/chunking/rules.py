"""Правила: один пункт (или подпункт, термин, таблица) = один чанк.

Чанк самодостаточен: в текст вшиты версия, часть, раздел и (для подпунктов) начало родительского
пункта, иначе фраза «Если стоимость имеет переменную величину Х...» не привязана ни к чему.
Слишком длинные пункты режутся по абзацам/предложениям, каждый кусок получает тот же заголовок.
"""
from __future__ import annotations

import re

from ..config import RULES_MAX_CHARS, RULES_PARENT_CONTEXT_CHARS
from ..normalize.rules import RuleItem
from ..schema import Chunk, ChunkMetadata

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


def _split_body(lines: list[str], limit: int) -> list[list[str]]:
    """Группирует строки в куски не длиннее limit; слишком длинную строку режет по предложениям."""
    units: list[str] = []
    for line in lines:
        if len(line) <= limit:
            units.append(line)
            continue
        buf = ""
        for sent in _SENT_SPLIT.split(line):
            if buf and len(buf) + 1 + len(sent) > limit:
                units.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            units.append(buf)
    groups: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for unit in units:
        if cur and size + len(unit) + 1 > limit:
            groups.append(cur)
            cur, size = [], 0
        cur.append(unit)
        size += len(unit) + 1
    if cur:
        groups.append(cur)
    return groups


def _section_path(item: RuleItem) -> str:
    parts = [f"Часть {item.part_no}"]
    if item.section_id != item.part_no:
        parts.append(item.section)
    if item.subsection:
        parts.append(item.subsection)
    parts.extend(item.path_ids)
    return " > ".join(parts)


def chunk_rules(
    items: list[RuleItem],
    *,
    doc_id: str,
    version: str,
    snapshot_date: str,
    max_chars: int = RULES_MAX_CHARS,
) -> list[Chunk]:
    by_id = {i.point_id: i for i in items}
    chunks: list[Chunk] = []

    for item in items:
        header = [f"Правила ККИ «Берсерк» {version} | {item.part}"]
        if item.section != item.part:
            header.append(item.section + (f" | {item.subsection}" if item.subsection else ""))
        elif item.subsection:
            header.append(item.subsection)
        parent = by_id.get(item.parent_id) if item.parent_id else None
        is_caption = item.kind == "table" and parent is not None and len(parent.lines[0]) <= 80
        if parent and parent.lines and not is_caption:
            header.append(f"Родительский пункт {parent.point_id}: {_truncate(parent.lines[0], RULES_PARENT_CONTEXT_CHARS)}")

        lines = [part for line in item.lines for part in line.split("\n") if part.strip()]
        if is_caption and lines[0] == "Таблица":      # подпись таблицы = короткий родительский пункт
            lines[0] = parent.lines[0].rstrip(".")
        repeat_first = item.kind == "table"           # у таблицы заголовок повторяется в каждом куске
        if item.kind == "point":
            lines[0] = f"{item.point_id}. {lines[0]}"

        budget = max(200, max_chars - sum(len(h) + 1 for h in header))
        if repeat_first and len(lines) > 1:
            groups = [[lines[0]] + g for g in _split_body(lines[1:], budget - len(lines[0]) - 1)]
        else:
            groups = _split_body(lines, budget)

        total = len(groups)
        for n, group in enumerate(groups, start=1):
            text_lines = list(header)
            if total > 1:
                text_lines.append(f"(часть {n} из {total})")
            text_lines.extend(group)
            suffix = f"#{n}" if total > 1 else ""
            chunks.append(Chunk(
                text="\n".join(text_lines),
                metadata=ChunkMetadata(
                    chunk_id=f"{doc_id}_{item.point_id}{suffix}",
                    doc_id=doc_id,
                    source_type="rules",
                    priority="official",
                    source_version=version,
                    snapshot_date=snapshot_date,
                    part=item.part,
                    section=item.section,
                    section_id=item.point_id,
                    section_path=_section_path(item),
                    parent_id=item.parent_id,
                ),
            ))
    return chunks
