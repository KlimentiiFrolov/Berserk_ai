"""Пути и константы. Ничего не создаётся при импорте (директории — по требованию)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("BERSERK_DATA_DIR", PROJECT_ROOT / "data"))
SNAPSHOT_DIR = DATA_DIR / "snapshots"   # зафиксированные входные данные (коммитятся в репо)
BUILD_DIR = DATA_DIR / "build"          # результаты прогонов: chunks.jsonl, manifest.json (в .gitignore)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_EMBEDDER = os.getenv("BERSERK_EMBEDDER", "e5-large-bm25")
DEFAULT_COLLECTION = os.getenv("BERSERK_COLLECTION", "")  # пусто -> имя строится из эмбеддера
LANGUAGE = "ru"

# Предельный размер текста чанка правил (символы). e5 режет вход на 512 токенах,
# для русского это примерно 1500-2000 символов; берём с запасом.
RULES_MAX_CHARS = 1600
RULES_PARENT_CONTEXT_CHARS = 180

# Карта «хэш картинки -> название символа». Заполняется командой вручную (см. docs/SCHEMA.md).
SYMBOLS_FILE = DATA_DIR / "symbols.json"


@dataclass(frozen=True)
class FaqSheet:
    key: str            # короткий ключ, идёт в doc_id: faq_<key>
    spreadsheet_id: str
    gid: str
    set_name: str       # имя вкладки как в таблице, например "4. Ложные боги"

    @property
    def csv_url(self) -> str:
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/export?format=csv&gid={self.gid}"

    @property
    def gviz_csv_url(self) -> str:
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/gviz/tq?tqx=out:csv&gid={self.gid}"

    @property
    def edit_url(self) -> str:
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/edit?gid={self.gid}"


FAQ_SPREADSHEET_ID = "1hSmaM_1SzxaWbWaIkPN4sgobjLWoACD97W8H_nhQ4WY"

# Известные вкладки FAQ. Для Sprint 1 заведена одна; остальные добавляются сюда
# (gid берётся из адресной строки при открытии вкладки) или передаются через CLI.
FAQ_SHEETS: dict[str, FaqSheet] = {
    "4": FaqSheet("4", FAQ_SPREADSHEET_ID, "522308198", "4. Ложные боги"),
}
