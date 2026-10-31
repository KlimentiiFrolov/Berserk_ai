# Sprint 2 — QA Robustness Extension

## Goal

Расширить QA-пакет после базового seed dataset v1.0 без зависимости от готовности интеграционного контура.

## Added

- `datasets/robustness/paraphrase_pairs.jsonl` — 10 перефразированных вариантов существующих gold-тестов.
- `datasets/robustness/false_premise.jsonl` — 8 сценариев с ложной предпосылкой или намеренно неверным фактом.
- `evaluation/error_taxonomy.json` — единая классификация типов ошибок для будущих прогонов.
- `docs/QA_COVERAGE_REPORT.md` — анализ текущего QA-покрытия, пробелов и заблокированных проверок.
- `scripts/validate_robustness.py` — автоматическая структурная проверка robustness dataset.

## Evaluation focus

### Semantic consistency
Система должна сохранять фактический смысл ответа при изменении формулировки вопроса.

### False-premise resistance
Система не должна принимать ложное утверждение пользователя как факт. Ожидается явное исправление неверной предпосылки с опорой на официальный источник.

### Error taxonomy
После появления стабильного RAG/e2e-контура ошибки можно классифицировать единообразно: factual mismatch, hallucination, retrieval miss, wrong source, incomplete answer, failed abstention и др.

## Sprint result

Базовый QA seed из Sprint 1 дополнен отдельным robustness-слоем. Новый набор не заменяет canonical `datasets/gold/all.jsonl` и не изменяет immutable gold v1.0.

Полный smoke/e2e и расчёт model-quality метрик остаются заблокированы до появления стабильного интеграционного контура.
