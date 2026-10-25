# berserk_ingest — ingestion-пайплайн «Берсерк AI»

`загрузка → нормализация → chunking → metadata → Qdrant (dense + BM25)`

Источники: правила (`.docx`) и FAQ (Google Sheets). Схема и решения: [`docs/SCHEMA.md`](docs/SCHEMA.md).

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # + requirements-optional.txt для e5-base
docker compose up -d qdrant

cp /путь/к/Расширенные_Правила_….docx data/snapshots/rules_8.0-1.docx
python -m berserk_ingest chunks --source faq --refresh     # скачать FAQ в снапшот (первый раз)
python -m berserk_ingest chunks --source all               # чанки без эмбеддингов: быстро, для проверки
python -m berserk_ingest reindex --source all --embedder e5-large-bm25 --collection berserk_test
python -m berserk_ingest search "нужно ли вскрывать Мираж" --collection berserk_test
```

Команды (`python -m berserk_ingest <cmd> -h`):

| Команда | Что делает |
|---|---|
| `chunks` | load → normalize → chunk, пишет `data/build/{chunks.jsonl,reports.json,manifest.json}`; Qdrant и модели не нужны |
| `reindex` | то же + эмбеддинги + заливка в Qdrant + сверка числа точек + smoke-поиск + манифест |
| `search` | запрос к коллекции (`--mode hybrid\|dense\|sparse`, фильтры `--source-type`, `--priority`) |
| `symbols` | выгрузить картинки-символы и шаблон для имён (`data/symbols.json`) |

Без сервера: `--local :memory:` (или путь к папке) запускает встроенный Qdrant из `qdrant-client`.

## Воспроизводимость

- Вход — **снапшоты** в `data/snapshots/` (см. там README); к Google Sheets ходит только `--refresh`.
- Чанки детерминированы: одинаковые входные файлы → побайтно одинаковый `chunks.jsonl` (хеш в манифесте, `chunks_sha256`).
- `reindex` по умолчанию пересоздаёт коллекцию; `--no-recreate` — синхронизация: точки тех же `doc_id` удаляются и заливаются заново.
- Манифест прогона (`data/build/manifest.json`): хеши входов, число чанков по источникам и приоритетам, модели и версии библиотек, число точек в коллекции, результаты smoke.
- Smoke-запросы: `data/smoke_queries.json` (ожидаемый чанк должен попасть в топ-3). `--strict` → код выхода 1 при провале.

## Смена модели (для сравнения качества)

`embedder` — любая модель с интерфейсом FastEmbed (`.embed(texts)`); BM25 передаётся отдельно как `sparse_embedder`:

```python
from fastembed import SparseTextEmbedding, TextEmbedding
from berserk_ingest.indexer import Loader
from berserk_ingest.pipeline import build

result, _ = build(["faq", "rules"])
loader = Loader()                                    # QdrantClient(url=QDRANT_URL), можно Loader(client)
loader.load_data(
    result.chunks,
    embedder=TextEmbedding("intfloat/multilingual-e5-large"),
    sparse_embedder=SparseTextEmbedding("Qdrant/bm25", language="russian"),   # не передавать = только dense
)
```

Имя коллекции по умолчанию строится из имён моделей, поэтому разные модели лежат в разных коллекциях.
Готовый пример сравнения с hit@3: `examples/compare_models.py`. Из CLI:
`python -m berserk_ingest reindex --dense-model intfloat/multilingual-e5-large --sparse-model Qdrant/bm25`
(`--no-sparse` — только dense). Пресеты (`--embedder e5-large-bm25`, `get_embedder(...)`) работают как раньше;
e5-base идёт через sentence-transformers, потому что его нет в fastembed.

Префиксы e5 (`passage: ` / `query: `) подставляются автоматически по имени модели (для остальных моделей префикса нет),
поэтому запросы и документы нельзя посчитать по-разному. Другой префикс — `wrap_fastembed(model, passage_prefix=...)`.
Новая модель со своим интерфейсом — строка в `PRESETS` или класс с методами `embed_passages` / `embed_query`.
Имена векторов в коллекции: `dense`, `sparse` (на них опирается retrieval).

## Структура

```
berserk_ingest/
  config.py  schema.py  embedders.py  indexer.py  pipeline.py  symbols.py  __main__.py
  loaders/    faq_sheets.py (CSV-снапшот)   rules_docx.py (python-docx: стили, нумерация, таблицы, картинки)
  normalize/  text.py  faq.py  numbering.py (счётчики Word)  rules.py (структура пунктов + проверка ссылок)
  chunking/   faq.py  rules.py
tests/        unittest/pytest; фикстура FAQ на реальных строках таблицы
docs/SCHEMA.md   data/snapshots/   data/smoke_queries.json   notebooks/ (только эксперименты)
```

## Тесты

```bash
python -m pytest              # или: python -m unittest discover -s tests -t .
BERSERK_RULES_DOCX=/путь/rules.docx python -m pytest tests/test_rules_structure.py   # проверка на реальном docx
```

Тесты на реальных правилах включаются, если docx лежит в `data/snapshots/rules*.docx` или задан `BERSERK_RULES_DOCX`.
Тесты `LocalQdrantTests` включаются, если установлен `qdrant-client` (встроенный Qdrant, сервер не нужен).
