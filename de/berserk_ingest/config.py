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
    gid: str            # нужен только для ссылки на вкладку; "" если неизвестен (загрузка идёт по имени вкладки)
    set_name: str       # имя вкладки как в таблице, например "4. Ложные боги"

    @property
    def xlsx_url(self) -> str:
        """Вся таблица одним файлом: из неё берутся все вкладки по именам, gid не нужны."""
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/export?format=xlsx"

    @property
    def csv_url(self) -> str:
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/export?format=csv&gid={self.gid}"

    @property
    def gviz_csv_url(self) -> str:
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/gviz/tq?tqx=out:csv&gid={self.gid}"

    @property
    def edit_url(self) -> str:
        suffix = f"?gid={self.gid}" if self.gid else ""
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/edit{suffix}"


FAQ_SPREADSHEET_ID = "1hSmaM_1SzxaWbWaIkPN4sgobjLWoACD97W8H_nhQ4WY"

# Вкладки FAQ по выпускам. Ключ идёт в doc_id (faq_<ключ>), имя должно точно совпадать с именем вкладки в таблице.
# Новый выпуск = новая строка здесь. Служебные вкладки («Чек-лист на игру», «Важные турнирные моменты»)
# устроены иначе и сюда не входят.
FAQ_SHEETS: dict[str, FaqSheet] = {
    "1": FaqSheet("1", FAQ_SPREADSHEET_ID, "", "1. Война стихий"),
    "2": FaqSheet("2", FAQ_SPREADSHEET_ID, "", "2. Нашествие тьмы"),
    "3": FaqSheet("3", FAQ_SPREADSHEET_ID, "", "3. Семена раздора"),
    "4": FaqSheet("4", FAQ_SPREADSHEET_ID, "522308198", "4. Ложные боги"),
    "5": FaqSheet("5", FAQ_SPREADSHEET_ID, "", "5. Путь хранителей"),
    "6": FaqSheet("6", FAQ_SPREADSHEET_ID, "", "6. Призрачный легион"),
    "7": FaqSheet("7", FAQ_SPREADSHEET_ID, "", "7. Гибель богов"),
    "8": FaqSheet("8", FAQ_SPREADSHEET_ID, "", "8. Сердце роя"),
}
