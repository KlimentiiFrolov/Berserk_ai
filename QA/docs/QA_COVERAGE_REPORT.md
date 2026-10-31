# QA Coverage Report — Sprint 2

## Scope

Отчёт фиксирует текущее QA-покрытие Berserk AI после расширения seed dataset сценариями robustness testing.

## Covered areas

- Basic rules
- Mechanics / glossary
- Card facts
- Card FAQ / interactions
- Tournament formats
- Negative / out-of-scope queries
- Paraphrase robustness
- False-premise resistance
- Smoke scenarios
- Error classification

## Current QA assets

| Asset | Count / status |
|---|---:|
| Gold test cases | 50 |
| Gold categories | 6 |
| Paraphrase robustness cases | 10 |
| False-premise cases | 8 |
| Smoke scenarios | 10 |
| Error taxonomy classes | 9 |

## Coverage gained in Sprint 2

### Semantic consistency
Добавлены перефразированные варианты существующих gold-кейсов. Они предназначены для проверки того, сохраняет ли система одинаковый фактический смысл ответа при изменении формулировки вопроса.

### False-premise resistance
Добавлены запросы с намеренно неверной предпосылкой. Система должна исправить ошибочный факт пользователя, а не продолжать ответ на его основе.

### Unified failure classification
Подготовлена единая taxonomy ошибок для последующего анализа результатов прогонов.

## Current coverage gaps

На текущем этапе остаются непокрытыми или неисполняемыми:

- multi-hop reasoning;
- conflicting-source resolution;
- long-context queries;
- retrieval ranking quality;
- latency / performance;
- load testing;
- Telegram → backend → RAG → response full E2E;
- regression comparison между версиями модели.

## Blocked checks

Полный smoke/e2e-прогон и расчёт фактических model-quality метрик требуют стабильного интеграционного контура.

До появления рабочего контура нельзя корректно фиксировать:

- pass rate;
- factual accuracy;
- retrieval success rate;
- citation correctness rate;
- latency;
- regression delta.

## Next QA milestone

После стабилизации интеграции:

1. выполнить P0 smoke suite;
2. прогнать canonical gold dataset;
3. выполнить robustness dataset;
4. классифицировать ошибки через `error_taxonomy.json`;
5. рассчитать метрики по категориям;
6. сохранить baseline для будущего regression testing.

## Sprint 2 conclusion

QA-контур расширен от базового seed dataset к robustness-oriented evaluation. Дополнительно сформирован анализ покрытия и определены следующие точки роста QA после появления стабильной реализации.
