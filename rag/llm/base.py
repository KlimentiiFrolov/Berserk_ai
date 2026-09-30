from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# Общие ошибки: адаптеры переводят в них исключения своего провайдера,
# чтобы вызывающий код не зависел от того, какой провайдер выбран.
class LLMError(Exception):
    pass

class LLMTimeoutError(LLMError):
    pass

class LLMUnavailableError(LLMError):
    # провайдер недоступен: нет соединения, 5xx
    pass


class LLMClient(Protocol):
    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,  # None - значение из LLMSettings
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    async def close(self) -> None: ...
