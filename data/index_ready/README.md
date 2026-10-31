# Index-ready data

В этой директории находятся входные данные для текущего ingestion-пайплайна
и offline-валидации RAG.

- `authoritative_index_docs.jsonl` — authoritative-источники, которые
  распределяются между коллекциями `cards`, `rules` и `faq`;
- `community_index_docs.jsonl` — community-источники для коллекции
  `community`.

Файлы были перенесены из старой реализации проекта. Они уже подготовлены в
формате `index-ready`, поэтому старые скрипты preprocessing из `de/` и
`old_berserk_for_ac3err/` для загрузки в Qdrant не нужны.

Загрузка выполняется модулем [rag_data/ingest.py](../../rag_data/ingest.py), а
mapping QA source ID — модулем
[rag_data/source_mapping.py](../../rag_data/source_mapping.py).
