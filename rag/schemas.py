from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class AnswerGuardState(Enum):
    REFUSE = 0
    WARN = 1
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
    stage: str  # hybrid_rrf | reranked
    answer_guard_state: AnswerGuardState = field(default=AnswerGuardState.OK)
