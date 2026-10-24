"""FAQ: одна запись (вопрос-ответ или уточнение по карте) = один чанк."""
from __future__ import annotations

import re

from ..normalize.faq import FaqRecord
from ..normalize.text import short_hash
from ..schema import Chunk, ChunkMetadata


def release_name(set_name: str) -> str:
    """'4. Ложные боги' -> 'Ложные боги'."""
    return re.sub(r"^\d+\.\s*", "", set_name).strip()


def faq_text(rec: FaqRecord) -> str:
    # Название карты и выпуск вшиты в текст намеренно: эмбеддер не видит metadata,
    # а короткий ответ вроде «Да.» без названия карты бесполезен для поиска.
    parts = [f"FAQ: {release_name(rec.set_name)}", f"Карта: {rec.card_name}"]
    if rec.question:
        parts.append(f"Вопрос: {rec.question}")
    parts.append(f"Ответ: {rec.answer}")
    return "\n".join(parts)


def chunk_faq(records: list[FaqRecord], *, doc_id: str, snapshot_date: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for rec in records:
        base = short_hash(rec.set_name, rec.card_number, rec.card_name, rec.question, rec.answer)
        n = seen.get(base, 0)
        seen[base] = n + 1
        # id не зависит от номера строки: перестановка строк в таблице не меняет id.
        chunk_id = f"{doc_id}_{base}" + (f"_{n}" if n else "")
        chunks.append(Chunk(
            text=faq_text(rec),
            metadata=ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_type="faq",
                priority="official" if rec.approval == "approved" else "community",
                source_version=rec.set_name,
                snapshot_date=snapshot_date,
                card_name=rec.card_name,
                card_number=rec.card_number,
                set_name=rec.set_name,
                faq_kind=rec.kind,
                approval_status=rec.approval,
            ),
        ))
    return chunks
