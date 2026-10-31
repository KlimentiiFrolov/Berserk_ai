# Тестирование RAG

Инструкция описывает offline-проверку retrieval и Answer Guard на старых
`index-ready` данных. Код проекта и команды ниже не запускаются автоматически.

## 1. Что проверяется

Проверка разделена на два независимых результата:

1. Retrieval: найден ли правильный legacy-чанк и на какой позиции.
2. Answer Guard: разрешает ли система ответить на вопрос, когда контекст слабый
   или отсутствует.

Качество текста ответа LLM оценивается отдельно и в этот retrieval-прогон не
входит.

## 2. Подготовка окружения

Из корня репозитория активируйте окружение и установите проект:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Запустите Qdrant:

```powershell
docker run -d --name berserkai-qdrant -p 6333:6333 qdrant/qdrant
```

Если контейнер уже существует, повторно создавать его не нужно. Проверить
Qdrant можно через `http://localhost:6333/dashboard`.

## 3. Загрузка старых данных в четыре коллекции

Во время загрузки отображаются progress bars `tqdm`: чтение JSONL, обработка
коллекций и отправка батчей в Qdrant. После чтения печатается количество
документов в каждой коллекции, а после загрузки — фактическое число points.

Используются файлы:

```
data/index_ready/authoritative_index_docs.jsonl
data/index_ready/community_index_docs.jsonl
```

Команда ingestion:

```powershell
python -m rag_data.ingest --authoritative data/index_ready/authoritative_index_docs.jsonl --community data/index_ready/community_index_docs.jsonl --qdrant-url http://localhost:6333
```

По умолчанию создаются или пересоздаются коллекции:

```
cards, rules, faq, community
```

`ingest.py` использует старые тексты и стабильные `source_id`, но заново строит
dense-векторы моделью `intfloat/multilingual-e5-large` и sparse-векторы
`Qdrant/bm25`.

Для baseline нельзя незаметно менять модель, исходные файлы или режим
`--no-recreate`. Все параметры должны быть записаны в отчёт эксперимента.

## 4. Построение mapping источников

QA использует логические ID (например, `SRC-001` и `CARD-001`), а legacy-данные используют
ID вида `knowledge_card_*` и `knowledge_faq_*`. Построить mapping можно так:

```powershell
python -m rag_data.source_mapping --authoritative data/index_ready/authoritative_index_docs.jsonl --community data/index_ready/community_index_docs.jsonl --output logs/legacy_source_mapping.json
```

Файл содержит для каждого логического источника:

- `status`: `mapped`, `ambiguous`, `missing`, `unavailable`;
- `match_type`;
- коллекцию;
- список допустимых `chunk_id`;
- поясняющие `notes`.

Не удаляйте `ambiguous` и `missing` записи вручную. Они должны быть видны в результатах и
исключаться из агрегированных метрик только как unmapped cases.

## 5. Запуск retrieval evaluation

Canonical QA-набор находится в:

```
QA/datasets/gold/all.jsonl
```

Запуск:

```powershell
python -m evaluate.eval_retrieval --dataset QA/datasets/gold/all.jsonl --source-mapping logs/legacy_source_mapping.json --output logs/retrieval_eval_baseline.json
```

Основная логика находится в `evaluate/eval_retrieval.py`. Старые импорты из
`rag_data.evaluation` сохранены как compatibility-wrapper.

Для каждого вопроса evaluator:

1. выбирает коллекцию по QA-категории;
2. вызывает hybrid retrieval;
3. применяет RRF и reranking;
4. получает финальные `top_k` чанков;
5. ограничивает gold mapping той же коллекцией;
6. считает Recall@K, Precision@K и MRR;
7. отдельно определяет ожидаемое состояние Answer Guard.

Соответствие категорий:

```
Rules_Basic             -> rules
Mechanics_Glossary      -> rules
Cards_Facts             -> cards
Card_FAQ_Interactions   -> faq
Tournament_Formats      -> rules
Negative_OutOfScope     -> все коллекции
```

## 6. Интерпретация метрик

`Recall@K` показывает, попал ли правильный чанк в первые `K` результатов.
`Precision@K` показывает долю правильных чанков среди первых `K`.
`MRR` штрафует правильный результат за низкую позицию.

Не смешивайте в один aggregate:

- positive answerable cases;
- `Negative_OutOfScope`;
- cases со статусом `unmapped`.

`Tournament_Formats` и другие cases без legacy gold должны быть явно отражены
как unmapped, а не считаться ошибками retrieval.

## 7. Настройка retrieval

Базовые параметры передаются в `evaluate_retriever`:

```
dense_limit
sparse_limit
rrf_limit
rerank_limit
top_k
use_reranker
```

Сначала зафиксируйте baseline:

```
dense_limit=20
sparse_limit=20
rrf_limit=20
rerank_limit=20
top_k=5
use_reranker=true
```

Затем меняйте одну группу параметров за раз и сохраняйте отдельный JSON-отчёт.
Минимальная таблица экспериментов:

| Эксперимент | dense | sparse | RRF | rerank | top_k |
|---|---:|---:|---:|---:|---:|
| baseline | 20 | 20 | 20 | 20 | 5 |
| wider candidates | 50 | 50 | 50 | 20 | 5 |
| wider rerank | 50 | 50 | 50 | 50 | 5 |
| no reranker | 50 | 50 | 50 | 0 | 5 |
| larger context | 50 | 50 | 50 | 20 | 8 |

Сравнивайте как минимум Recall@1, Recall@3, Recall@5, Precision@5 и MRR по
каждой QA-категории.

## 8. Калибровка Answer Guard

Пороговые значения находятся в `core/config.py`:

```
refuse_threshold = 0.1
warn_threshold = 0.5
```

Нужно проверить распределение `top_score` для:

- answerable вопросов с правильным retrieval;
- answerable вопросов с плохим retrieval;
- out-of-scope вопросов;
- пустого retrieval.

Правило принятия решения:

```
score < refuse_threshold          -> REFUSE
refuse_threshold <= score < warn -> WARN
score >= warn_threshold           -> OK
```

Порог нельзя переносить между режимами без повторной калибровки: score после
reranker и score без reranker имеют разный смысл.

## 9. Что считать хорошим результатом

Минимальная цель baseline:

- positive QA имеет высокий Recall@5;
- правильный чанк обычно находится в top-1/top-3;
- `Negative_OutOfScope` не получает состояние `OK` на слабом контексте;
- `unmapped` cases не портят агрегированные метрики;
- результаты воспроизводимы при повторном ingestion;
- `chunk_id` и collection совпадают с gold mapping.

Высокий Recall ещё не доказывает, что LLM сформировала правильный ответ. Для
этого нужен отдельный manual/LLM-judge прогон по `expected_facts`,
`expected_answer` и `actual_sources`, описанный в `QA/evaluation/README.md`.

## 10. Файлы результатов

Рекомендуемая структура:

```
logs/
├── legacy_source_mapping.json
├── retrieval_eval_baseline.json
├── retrieval_eval_wider_candidates.json
├── retrieval_eval_no_reranker.json
└── guard_thresholds.json
```

Gold QA-файлы не изменяйте во время экспериментов. Каждый прогон должен иметь
собственные параметры, mapping, дату и итоговый JSON-отчёт.
