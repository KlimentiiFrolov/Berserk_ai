"""Загрузка вкладки FAQ из Google Sheets.

Правило воспроизводимости: пайплайн читает *снапшот* (CSV + meta.json в data/snapshots).
К живой таблице он обращается только при первом запуске или с refresh=True.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import FaqSheet


@dataclass
class RawFaqSheet:
    sheet: FaqSheet
    rows: list[list[str]]
    sha256: str
    path: Path
    snapshot_date: str          # YYYY-MM-DD
    fetched_at: str | None


def snapshot_csv_path(sheet: FaqSheet, snapshot_dir: Path) -> Path:
    return snapshot_dir / f"faq_{sheet.key}.csv"


def _meta_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def _looks_like_html(data: bytes, content_type: str) -> bool:
    prefix = data[:200].lstrip().lower()
    return "text/html" in content_type.lower() or prefix.startswith((b"<!doctype html", b"<html"))


def fetch_csv(sheet: FaqSheet, *, timeout: int = 120) -> tuple[bytes, str]:
    """Скачивает CSV вкладки. Возвращает (байты, использованный URL)."""
    errors: list[str] = []
    for url in (sheet.csv_url, sheet.gviz_csv_url):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 BerserkIngest/0.1", "Accept": "*/*"})
            with urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                ctype = resp.headers.get("content-type", "")
            if _looks_like_html(data, ctype):
                errors.append(f"{url}: вместо CSV пришёл HTML (нет публичного доступа?)")
                continue
            return data, url
        except (HTTPError, URLError, TimeoutError) as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Не удалось скачать CSV вкладки:\n" + "\n".join(errors))


def parse_csv(text: str) -> list[list[str]]:
    # StringIO(newline="") обязателен: в ячейках бывают переводы строк внутри кавычек,
    # а text.splitlines() рвёт такие ячейки пополам.
    rows = [list(r) for r in csv.reader(io.StringIO(text, newline=""))]
    width = max((len(r) for r in rows), default=0)
    return [r + [""] * (width - len(r)) for r in rows]


def load_faq_sheet(
    sheet: FaqSheet,
    snapshot_dir: Path,
    *,
    refresh: bool = False,
    csv_path: Path | None = None,
) -> RawFaqSheet:
    path = Path(csv_path) if csv_path else snapshot_csv_path(sheet, snapshot_dir)
    meta_file = _meta_path(path)
    fetched_at: str | None = None

    if refresh or not path.exists():
        if csv_path and not path.exists():
            raise FileNotFoundError(f"Снапшот не найден: {path}")
        print(f"[faq] скачиваю вкладку {sheet.set_name!r} (gid={sheet.gid}) ...")
        data, used_url = fetch_csv(sheet)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta_file.write_text(json.dumps({
            "source_url": sheet.edit_url, "effective_url": used_url, "set_name": sheet.set_name,
            "fetched_at": fetched_at, "sha256": hashlib.sha256(data).hexdigest(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()

    snapshot_date = None
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        fetched_at = fetched_at or meta.get("fetched_at")
        if fetched_at:
            snapshot_date = fetched_at[:10]
    if snapshot_date is None:
        # без meta.json дата берётся из mtime файла и может «плыть» при git checkout
        snapshot_date = date.fromtimestamp(path.stat().st_mtime).isoformat()
        print(f"[faq] ВНИМАНИЕ: нет {meta_file.name}, snapshot_date взята из mtime ({snapshot_date})")

    rows = parse_csv(data.decode("utf-8-sig"))
    return RawFaqSheet(sheet, rows, sha, path, snapshot_date, fetched_at)
