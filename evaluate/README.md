# Оценка ответов LLM в RAG

## Что измеряется

Скрипт [eval_llm_answers.py](./eval_llm_answers.py) запускает полный
пользовательский контур:

`вопрос -> retrieval -> Answer Guard -> LLM -> sources`

Для каждого QA-кейса сохраняются:

- ответ модели;
- состояние Answer Guard;
- retrieved chunks и их scores;
- источники и chunk_id;
- expected answer/facts;
- deterministic-метрики;
- результат LLM-as-a-judge.

## Настройка API

Используется существующий клиент из [rag/llm/](../rag/llm/), а настройки читаются
из файла .env через [core/config.py](../core/config.py).

Для Ollama:

```dotenv
LLM__PROVIDER=ollama
LLM__BASE_URL=http://localhost:11434
LLM__MODEL=qwen2.5:7b-instruct
LLM__API_KEY=
```

Для OpenAI или OpenAI-compatible API:

```dotenv
LLM__PROVIDER=openai
LLM__BASE_URL=https://api.openai.com/v1
LLM__MODEL=<model-name>
LLM__API_KEY=<secret>
```

Основная модель и judge используют один настроенный API-клиентский слой.
Judge вызывается отдельным запросом с temperature=0.

## Запуск

Сначала должны быть доступны Qdrant и индексированные коллекции. Затем:

```powershell
python -m evaluate.eval_llm_answers --dataset QA/datasets/gold/all.jsonl --source-mapping logs/legacy_source_mapping.json --output logs/llm_eval_baseline.json
```

Для проверки только deterministic-части без дополнительного judge-запроса:

```powershell
python -m evaluate.eval_llm_answers --dataset QA/datasets/gold/all.jsonl --source-mapping logs/legacy_source_mapping.json --output logs/llm_eval_deterministic.json --no-judge
```

Для smoke-прогона:

```powershell
python -m evaluate.eval_llm_answers --dataset QA/datasets/gold/all.jsonl --output logs/llm_eval_smoke.json --max-cases 3
```

`--no-judge` не отключает основную генерацию ответа; он отключает только
дополнительную проверку ответа через judge.

## Графики retrieval-валидации

`eval_retrieval` автоматически сохраняет подробные PNG-графики в каталог
`<имя_отчёта>_plots`. По умолчанию для отчёта
`logs/retrieval_eval.json` это `logs/retrieval_eval_plots`.

Сохраняются:

- `aggregate_metrics.png` — средние Recall@K, Precision@K и MRR;
- `metrics_by_category.png` — метрики по QA-категориям;
- `recall_precision_by_k.png` — зависимость Recall и Precision от K;
- `mrr_distribution.png` — распределение MRR по кейсам;
- `top_score_guard_thresholds.png` — распределение верхнего retrieval score
  и пороги Answer Guard.

Для другого каталога можно передать `--plots-dir`. Изображения сохраняются в
PNG с `dpi=300` и `bbox_inches="tight"`, поэтому их можно увеличивать для
детального анализа.

Для LLM-валидации команда `eval_llm_answers` также автоматически сохраняет
графики в `<имя_отчёта>_plots`:

- `deterministic_metrics.png` — Guard accuracy, source/citation hit rate и
  non-empty rate;
- `judge_mean_scores.png` — средние оценки Judge;
- `judge_labels.png` — количество PASS, PARTIAL и FAIL;
- `answer_guard_states.png` — распределение состояний Answer Guard;
- `judge_score_distribution.png` — распределение Judge score;
- `llm_metrics_by_category.png` — метрики по QA-категориям.

## Deterministic-метрики

### Guard accuracy

Проверяет ожидаемое поведение Answer Guard:

- answerable case: состояние не должно быть `REFUSE`;
- unanswerable case: ожидается `REFUSE`.

Это не оценка фактической правильности ответа. Ответ может пройти Guard и
содержать ошибку.

### Source hit rate

Проверяет, попал ли хотя бы один retrieved chunk в gold mapping с учётом
коллекции.

### Citation hit rate

Проверяет, присутствует ли gold chunk среди источников, прикреплённых к ответу.
В текущей архитектуре sources строятся из retrieved chunks детерминированно,
поэтому эта метрика контролирует корректность source builder и provenance.

## LLM-as-a-judge

Judge получает:

- вопрос;
- expected_answer;
- expected_facts;
- forbidden_claims;
- ответ RAG;
- состояние Guard;
- контекст retrieved chunks.

Он возвращает:

- label: PASS, PARTIAL или FAIL;
- score: от 0 до 100;
- factual_correctness;
- faithfulness;
- answer_relevance;
- abstention_quality;
- покрытые и пропущенные факты;
- unsupported claims;
- комментарий.

Агрегированные показатели:

- pass_rate;
- partial_rate;
- fail_rate;
- средний score;
- средние значения каждого критерия.

Judge — вероятностная оценка, поэтому при сравнении моделей фиксируйте:

- QA snapshot;
- модель и provider;
- temperature;
- max tokens;
- дату прогона;
- prompt judge.

## Как интерпретировать результат

Для хорошего RAG должны одновременно выполняться условия:

1. высокий source/retrieval hit rate;
2. высокая factual correctness;
3. высокая faithfulness;
4. низкий fail rate;
5. корректный отказ на unanswerable;
6. отсутствие необъяснимого расхождения между retrieved sources и citations.

Нельзя использовать только BERTScore или только средний judge score:

- BERTScore измеряет текстовую похожесть, но может пропустить неправильное число;
- judge может ошибаться и должен рассматриваться как дополнительный сигнал;
- retrieval hit не доказывает, что LLM правильно использовала контекст;
- Guard accuracy не доказывает factual correctness.

## Файл отчёта

Рекомендуемая структура:

```
logs/
├── llm_eval_baseline.json
├── llm_eval_deterministic.json
└── llm_eval_smoke.json
```
