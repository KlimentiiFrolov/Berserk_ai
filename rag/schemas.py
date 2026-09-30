from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class AnswerGuardState(Enum):
    REFUSE = 0  # релевантного контекста нет - отказ без вызова LLM
    WARN = 1    # контекст слабый - ответ с предупреждением
    OK = 2

@dataclass
class Chunk:
    id: str
    text: str
    source_type: str
    collection: str
    metadata: dict = field(default_factory=dict)

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    stage: Literal["hybrid_rrf", "reranked"]

@dataclass
class Source:
    # Строится из payload чанка, без участия LLM
    collection: str
    source_type: Literal["official", "community"]
    title: str # пункт правил / название карты / вопрос FAQ / заголовок статьи
    url: str | None = None

@dataclass
class Answer:
    text: str
    guard_state: AnswerGuardState
    sources: list[Source] = field(default_factory=list)
    routed_collections: list[str] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)  # для логов и eval
