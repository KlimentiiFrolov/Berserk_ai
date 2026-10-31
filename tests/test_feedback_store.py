import asyncio
import json

import aiosqlite
import pytest

from rag.schemas import Source
from storage.feedback import FeedbackStore


async def rows(path, query):
    async with aiosqlite.connect(path) as db:
        async with db.execute(query) as cursor:
            return await cursor.fetchall()


async def test_persistence_upsert_and_independent_users(tmp_path):
    path = tmp_path / "data" / "feedback.sqlite3"
    store = FeedbackStore(path)
    await store.initialize()
    answer_id = await store.save_answer("Вопрос ' ?", "Ответ 🃏", -42)
    assert await store.save_rating(answer_id, 7, 1, -42)
    assert await store.save_rating(answer_id, 7, 1, -42)
    assert await store.save_rating(answer_id, 7, -1, -42)
    # Новое соединение/объект не теряет прежние данные.
    reopened = FeedbackStore(path)
    await reopened.initialize()
    assert await reopened.save_rating(answer_id, 8, 1, -42)
    assert await rows(path, "SELECT user_id, rating FROM feedback ORDER BY user_id") == [
        (7, -1),
        (8, 1),
    ]
    saved = await rows(path, "SELECT question, answer, chat_id, created_at FROM answers")
    assert saved[0][:3] == ("Вопрос ' ?", "Ответ 🃏", -42)
    assert saved[0][3].endswith("Z")


async def test_missing_answer_wrong_chat_and_invalid_rating(tmp_path):
    path = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(path)
    await store.initialize()
    answer_id = await store.save_answer("Вопрос", "Ответ", 42)
    assert not await store.save_rating(answer_id + 1, 7, 1, 42)
    assert not await store.save_rating(answer_id, 7, 1, 99)
    with pytest.raises(ValueError):
        await store.save_rating(answer_id, 7, 0, 42)
    assert await rows(path, "SELECT * FROM feedback") == []


async def test_concurrent_votes_do_not_duplicate_rows(tmp_path):
    path = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(path)
    await store.initialize()
    answer_id = await store.save_answer("Вопрос", "Ответ", 42)
    results = await asyncio.gather(*(store.save_rating(answer_id, 7, 1, 42) for _ in range(8)))
    assert all(results)
    assert await rows(path, "SELECT COUNT(*) FROM feedback") == [(1,)]


async def test_sources_migration_preserves_existing_answers_and_ratings(tmp_path):
    path = tmp_path / "old.sqlite3"
    async with aiosqlite.connect(path) as db:
        await db.executescript("""
            CREATE TABLE answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL,
                answer TEXT NOT NULL, chat_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE feedback (
                answer_id INTEGER NOT NULL REFERENCES answers(id), user_id INTEGER NOT NULL,
                rating INTEGER NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (answer_id, user_id)
            );
            INSERT INTO answers (question, answer, chat_id) VALUES ('вопрос', 'ответ', 42);
            INSERT INTO feedback (answer_id, user_id, rating) VALUES (1, 7, -1);
        """)
        await db.commit()
    store = FeedbackStore(path)
    await store.initialize()
    await store.initialize()
    assert await rows(path, "SELECT id, answer, sources_json FROM answers") == [(1, "ответ", "[]")]
    assert await rows(path, "SELECT answer_id, user_id, rating FROM feedback") == [(1, 7, -1)]
    sources = [
        Source("cards", "official", "Карта: Пример"),
        Source("community", "community", "Статья", "https://example.org/article"),
    ]
    new_id = await store.save_answer("новый вопрос", "новый ответ", 42, sources=sources)
    saved = await rows(path, "SELECT id, sources_json FROM answers ORDER BY id")
    assert saved[1][0] == new_id == 2
    assert json.loads(saved[1][1]) == [
        {"collection": "cards", "source_type": "official", "title": "Карта: Пример", "url": None},
        {
            "collection": "community",
            "source_type": "community",
            "title": "Статья",
            "url": "https://example.org/article",
        },
    ]
