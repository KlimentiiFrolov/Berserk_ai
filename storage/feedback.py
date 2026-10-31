import json
from dataclasses import asdict
from pathlib import Path

import aiosqlite

from rag.schemas import Source


class FeedbackStore:
    """Отдельное соединение на операцию; изменения фиксируются до возврата."""

    def __init__(self, path: Path):
        self.path = path

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    chat_id INTEGER NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    answer_id INTEGER NOT NULL REFERENCES answers(id),
                    user_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    PRIMARY KEY (answer_id, user_id)
                );
            """)
            # Обновить существующую БД без потери ответов и оценок.
            await db.execute("BEGIN IMMEDIATE")
            async with db.execute("PRAGMA table_info(answers)") as cursor:
                columns = {row[1] for row in await cursor.fetchall()}
            if "sources_json" not in columns:
                await db.execute(
                    "ALTER TABLE answers ADD COLUMN sources_json TEXT NOT NULL DEFAULT '[]'"
                )
            await db.commit()

    async def save_answer(
        self, question: str, answer: str, chat_id: int, *, sources: list[Source] | None = None
    ) -> int:
        sources_json = json.dumps([asdict(source) for source in sources or []], ensure_ascii=False)
        async with aiosqlite.connect(self.path) as db:
            async with db.execute(
                "INSERT INTO answers (question, answer, chat_id, sources_json) VALUES (?, ?, ?, ?)",
                (question, answer, chat_id, sources_json),
            ) as cursor:
                answer_id = cursor.lastrowid
            await db.commit()
        assert answer_id is not None
        return answer_id

    async def save_rating(self, answer_id: int, user_id: int, rating: int, chat_id: int) -> bool:
        """Upsert оценки; False, если ответ отсутствует или принадлежит другому чату."""
        if rating not in (-1, 1):
            raise ValueError("Оценка должна быть 1 или -1")
        async with aiosqlite.connect(self.path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            async with db.execute(
                """INSERT INTO feedback (answer_id, user_id, rating)
                   SELECT id, ?, ? FROM answers WHERE id = ? AND chat_id = ?
                   ON CONFLICT(answer_id, user_id) DO UPDATE SET
                       rating = excluded.rating,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')""",
                (user_id, rating, answer_id, chat_id),
            ) as cursor:
                saved = cursor.rowcount == 1
            await db.commit()
        return saved
