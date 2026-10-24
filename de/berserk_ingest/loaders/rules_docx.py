"""Загрузка правил из .docx через python-docx: текст, стили, нумерация, таблицы, картинки-символы.

Loader ничего не «понимает» про смысл правил: он отдаёт абзацы в порядке документа
вместе с сырыми признаками нумерации (numId, ilvl) и определениями нумерации из numbering.xml.
Сборка номеров пунктов (105.2.3) — в normalize/rules.py.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator, Literal

import docx
from docx.table import Table

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


@dataclass(frozen=True)
class NumLevel:
    fmt: str        # decimal | lowerLetter | upperLetter | lowerRoman | upperRoman | bullet | ...
    text: str       # lvlText, например "105.%1." или "%2." или "●"
    start: int = 1


@dataclass
class RawParagraph:
    index: int                      # позиция в теле документа
    kind: Literal["p", "table"]
    style: str
    text: str
    num_id: int | None = None
    ilvl: int | None = None


@dataclass
class SymbolInfo:
    key: str                        # sha1(картинка)[:10]
    ext: str
    blob: bytes
    count: int = 0
    context: str = ""               # фрагмент текста вокруг первого вхождения
    name: str | None = None         # человекочитаемое имя из symbols.json


@dataclass
class RawRules:
    path: Path
    sha256: str
    title: str
    version: str
    snapshot_date: str
    paragraphs: list[RawParagraph]
    numbering: dict[int, dict[int, NumLevel]]
    symbols: dict[str, SymbolInfo] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _read_numbering(document: "docx.document.Document") -> dict[int, dict[int, NumLevel]]:
    try:
        root = document.part.numbering_part.element
    except (KeyError, NotImplementedError):  # в документе нет нумерации
        return {}
    abstract: dict[str, dict[int, NumLevel]] = {}
    for a in root.findall(W + "abstractNum"):
        levels: dict[int, NumLevel] = {}
        for lvl in a.findall(W + "lvl"):
            fmt = lvl.find(W + "numFmt")
            txt = lvl.find(W + "lvlText")
            st = lvl.find(W + "start")
            levels[int(lvl.get(W + "ilvl"))] = NumLevel(
                fmt=fmt.get(W + "val") if fmt is not None else "decimal",
                text=txt.get(W + "val") if txt is not None else "",
                start=int(st.get(W + "val")) if st is not None else 1,
            )
        abstract[a.get(W + "abstractNumId")] = levels

    result: dict[int, dict[int, NumLevel]] = {}
    for n in root.findall(W + "num"):
        levels = dict(abstract.get(n.find(W + "abstractNumId").get(W + "val"), {}))
        for ov in n.findall(W + "lvlOverride"):          # переопределение начала нумерации
            so = ov.find(W + "startOverride")
            ilvl = int(ov.get(W + "ilvl"))
            if so is not None and ilvl in levels:
                levels[ilvl] = NumLevel(levels[ilvl].fmt, levels[ilvl].text, int(so.get(W + "val")))
        result[int(n.get(W + "numId"))] = levels
    return result


class _TextExtractor:
    """Текст абзаца в порядке документа. Картинки заменяются на [символ] / [символ: имя]."""

    def __init__(self, document: "docx.document.Document", symbol_names: dict[str, str]):
        self.document = document
        self.symbol_names = symbol_names
        self.symbols: dict[str, SymbolInfo] = {}

    def _symbol(self, rid: str) -> tuple[str, str]:
        part = self.document.part.related_parts[rid]
        key = hashlib.sha1(part.blob).hexdigest()[:10]
        info = self.symbols.get(key)
        if info is None:
            ext = Path(str(part.partname)).suffix.lstrip(".") or "bin"
            info = self.symbols[key] = SymbolInfo(key, ext, part.blob, name=self.symbol_names.get(key))
        info.count += 1
        return key, (f"[символ: {info.name}]" if info.name else "[символ]")

    def _walk(self, el, out: list[str], keys: list[str]) -> None:
        for child in el:
            tag = child.tag
            if tag == W + "t":
                out.append(child.text or "")
            elif tag == W + "tab":
                out.append(" ")
            elif tag in (W + "br", W + "cr"):
                out.append("\n")
            elif tag == W + "drawing":
                for blip in child.iter(A + "blip"):
                    rid = blip.get(R + "embed")
                    if rid:
                        key, token = self._symbol(rid)
                        keys.append(key)
                        out.append(f" {token} ")
            elif tag.endswith("}Fallback"):   # дубль содержимого mc:AlternateContent
                continue
            else:
                self._walk(child, out, keys)

    def paragraph(self, p_el) -> tuple[str, list[str]]:
        out: list[str] = []
        keys: list[str] = []
        self._walk(p_el, out, keys)
        text = "".join(out)
        text = re.sub(r"[ \t ]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text).strip()
        text = re.sub(r"\s+([,.;:!?)])", r"\1", text)      # пробел перед знаком, оставшийся от плейсхолдера
        return text, keys


def _serialize_table(tbl_el, parent, ext: _TextExtractor) -> str:
    """Каждая строка таблицы -> самодостаточная строка «метка | колонка: значение; ...»."""
    table = Table(tbl_el, parent)
    matrix: list[list[str]] = []
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:
            if id(cell._tc) in seen:        # объединённая ячейка встречается несколько раз
                continue
            seen.add(id(cell._tc))
            parts = [ext.paragraph(p)[0] for p in cell._tc.iter(W + "p")]
            cells.append(" ".join(t for t in parts if t))
        matrix.append(cells)
    if not matrix:
        return ""
    header = matrix[0]
    lines: list[str] = []
    for cells in matrix[1:]:
        if len(cells) == len(header):
            label = cells[0]
            rest = "; ".join(f"{h}: {v}" for h, v in zip(header[1:], cells[1:]) if v)
            lines.append(f"{label} | {rest}" if label else rest)
        else:   # строка с объединёнными ячейками
            lines.append(f"{cells[0]}: {' '.join(cells[1:])}" if len(cells) > 1 else cells[0])
    return "\n".join(l for l in lines if l.strip())


_TITLE_RE = re.compile(r"версия\s+([0-9][0-9.\-]*)\s+от\s+(\d{2})\.(\d{2})\.(\d{4})", re.IGNORECASE)


def load_rules_docx(path: Path, symbol_names: dict[str, str] | None = None) -> RawRules:
    path = Path(path)
    data = path.read_bytes()
    document = docx.Document(path)
    ext = _TextExtractor(document, symbol_names or {})
    numbering = _read_numbering(document)

    paragraphs: list[RawParagraph] = []
    for index, el in enumerate(document.element.body.iterchildren()):
        if el.tag == W + "p":
            ppr = el.find(W + "pPr")
            style_el = ppr.find(W + "pStyle") if ppr is not None else None
            style_id = style_el.get(W + "val") if style_el is not None else "normal"
            style = next((s.name for s in document.styles if s.style_id == style_id), style_id)
            num_id = ilvl = None
            numpr = ppr.find(W + "numPr") if ppr is not None else None
            if numpr is not None:
                n, l = numpr.find(W + "numId"), numpr.find(W + "ilvl")
                if n is not None and int(n.get(W + "val")) != 0:
                    num_id = int(n.get(W + "val"))
                    ilvl = int(l.get(W + "val")) if l is not None else 0
            text, keys = ext.paragraph(el)
            for k in keys:
                if not ext.symbols[k].context:
                    ext.symbols[k].context = text[:140]
            paragraphs.append(RawParagraph(index, "p", style, text, num_id, ilvl))
        elif el.tag == W + "tbl":
            paragraphs.append(RawParagraph(index, "table", "table", _serialize_table(el, document._body, ext)))

    title = next((p.text for p in paragraphs if p.style == "Title"), "")
    warnings: list[str] = []
    m = _TITLE_RE.search(title)
    if m:
        version = m.group(1)
        snapshot_date = f"{m.group(4)}-{m.group(3)}-{m.group(2)}"
    else:
        version = "unknown"
        snapshot_date = date.fromtimestamp(path.stat().st_mtime).isoformat()
        warnings.append("не удалось извлечь версию/дату из заголовка документа; дата взята из mtime файла")

    return RawRules(
        path=path, sha256=hashlib.sha256(data).hexdigest(), title=title, version=version,
        snapshot_date=snapshot_date, paragraphs=paragraphs, numbering=numbering,
        symbols=ext.symbols, warnings=warnings,
    )
