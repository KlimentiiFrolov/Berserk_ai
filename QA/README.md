# Berserk AI — QA / Evaluation

QA-пакет для MVP RAG-ассистента по ККИ «Берсерк».

## Structure

```text
qa/
├── README.md
├── datasets/
│   ├── test_index.csv
│   └── gold/
│       ├── all.jsonl
│       ├── rules_basic.jsonl
│       ├── mechanics_glossary.jsonl
│       ├── cards_facts.jsonl
│       ├── card_faq_interactions.jsonl
│       ├── tournament_formats.jsonl
│       └── negative_out_of_scope.jsonl
├── sources/
│   ├── official_sources.json
│   ├── official_sources.csv
│   └── SOURCE_POLICY.md
├── schemas/
│   ├── test_case.schema.json
│   └── source.schema.json
├── evaluation/
│   ├── rubric.json
│   └── README.md
├── smoke/
│   ├── smoke_tests.json
│   └── README.md
├── docs/
│   └── DATASET_CARD.md
├── scripts/
│   └── validate_dataset.py
└── exports/
    └── berserk_ai_qa_seed_v1.xlsx

```

## Canonical files

- `datasets/gold/all.jsonl` — главный gold dataset.
- Файлы внутри `datasets/gold/` по категориям — представления того же набора для удобства командной работы.
- `sources/official_sources.json` — реестр официальных источников.
- `schemas/` — контракт данных.
- `evaluation/rubric.json` — правила оценки ответов.
- `smoke/smoke_tests.json` — smoke suite MVP.
- `exports/*.xlsx` — только человекочитаемый экспорт; **не canonical storage**.

## Important

Не редактировать одновременно `all.jsonl` и category-файлы независимо. Источник истины для v1.0 — `all.jsonl`.
При изменении dataset лучше обновить `all.jsonl`, затем регенерировать category-файлы.

## Quick validation

```bash
python QA/scripts/validate_dataset.py
```

Validator проверяет:
- уникальность `test_id`;
- наличие обязательных полей;
- существование всех `expected_source_ids`;
- соответствие category-файлов `all.jsonl`;
- наличие `snapshot_date` у temporal-тестов.
