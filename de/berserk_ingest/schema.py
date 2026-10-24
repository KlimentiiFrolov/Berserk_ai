"""Единый контракт чанка. И rules, и FAQ-чанкеры обязаны отдавать `Chunk`."""
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

SourceType = Literal["rules", "faq"]
Priority = Literal["official", "community"]
ApprovalStatus = Literal["approved", "not_approved", "unknown"]


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- общие поля (обязательны для всех источников) ---
    chunk_id: str                 # стабильный ключ чанка, из него строится id точки в Qdrant
    doc_id: str                   # rules_<версия> | faq_<ключ вкладки>
    source_type: SourceType
    priority: Priority            # official | community (для FAQ из «Ответ утвержден»)
    source_version: str           # rules: "8.0-1"; faq: имя вкладки ("4. Ложные боги")
    snapshot_date: str            # YYYY-MM-DD — дата версии источника, а не время запуска
    language: str = "ru"

    # --- только rules ---
    part: str | None = None           # "Часть 1. Основные принципы игры ..."
    section: str | None = None        # "105. Базовые свойства."
    section_id: str | None = None     # номер пункта: "105.2.3"
    section_path: str | None = None   # "Часть 1 > 105. Базовые свойства > 105.2 > 105.2.3"
    parent_id: str | None = None      # номер родительского пункта: "105.2"

    # --- только faq ---
    card_name: str | None = None
    card_number: int | None = None
    set_name: str | None = None
    faq_kind: Literal["qna", "ruling"] | None = None   # qna = вопрос+ответ, ruling = только ответ про карту
    approval_status: ApprovalStatus | None = None

    @model_validator(mode="after")
    def _check_source_specific(self) -> "ChunkMetadata":
        if self.source_type == "rules":
            if not (self.section_id and self.section_path):
                raise ValueError("rules-чанк требует section_id и section_path")
            if self.priority != "official":
                raise ValueError("правила всегда official")
        else:
            missing = [f for f in ("card_name", "set_name", "faq_kind", "approval_status")
                       if getattr(self, f) is None]
            if missing:
                raise ValueError(f"faq-чанк требует поля: {missing}")
        return self


class Chunk(BaseModel):
    text: str                     # именно этот текст эмбеддится и показывается в источниках
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id

    def payload(self) -> dict:
        """Payload точки в Qdrant: metadata + text."""
        return {**self.metadata.model_dump(exclude_none=True), "text": self.text}


def point_id(chunk_id: str) -> str:
    """Детерминированный UUID точки: повторная заливка перезаписывает, а не дублирует."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
