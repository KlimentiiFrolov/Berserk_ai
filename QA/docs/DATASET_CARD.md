# Dataset Card — Berserk AI QA Seed v1.0

## Purpose

Первичный gold-набор для QA и evaluation RAG-ассистента по ККИ «Берсерк».

- Version: `1.0`
- Snapshot date: `2026-10-03`
- Language: `ru-RU`
- Total tests: `50`
- Split: `gold_seed`
- Source policy: official sources only

## Category distribution

| Category | Count |
|---|---:|
| Rules_Basic | 15 |
| Mechanics_Glossary | 12 |
| Cards_Facts | 8 |
| Card_FAQ_Interactions | 8 |
| Tournament_Formats | 5 |
| Negative_OutOfScope | 2 |

## Difficulty

- easy: 14
- medium: 21
- hard: 15

## Answerability

- answerable: 48
- unanswerable: 2

## Design principles

1. Gold оценивается по фактам и источникам, а не по дословному совпадению.
2. У каждого теста есть ожидаемый retrieval target и source IDs.
3. `forbidden_claims` фиксирует критические галлюцинации/ошибки.
4. Динамические сведения имеют `snapshot_date`.
5. Negative tests проверяют корректный abstention.
6. Старые gold-версии не следует переписывать без изменения dataset version.

## Limitations of v1.0

- Это seed dataset, а не исчерпывающая выборка всех правил.
- Доля сложных multi-hop взаимодействий пока ограничена.
- Турнирные данные могут изменяться после snapshot date.
- Будущие версии стоит пополнять реальными пользовательскими вопросами и найденными production failure cases.
