from collections.abc import Callable

from core.config import LLMSettings
from rag.llm.base import (
    LLMClient,
    LLMError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
)
from rag.llm.limiter import ConcurrencyLimitedLLM
from rag.llm.ollama import OllamaClient
from rag.llm.openai_client import OpenAIClient

# Реестр провайдеров: ключ - значение LLM__PROVIDER.
# Новый провайдер: адаптер с интерфейсом LLMClient + запись здесь + значение в LLMSettings.provider
LLM_PROVIDERS: dict[str, Callable[[LLMSettings], LLMClient]] = {
    "ollama": OllamaClient,
    "openai": OpenAIClient,
}


def create_llm(settings: LLMSettings) -> ConcurrencyLimitedLLM:
    """Создаёт клиент по settings.provider, обёрнутый в очередь генерации."""
    client = LLM_PROVIDERS[settings.provider](settings)
    return ConcurrencyLimitedLLM(client, settings.max_concurrency)


__all__ = [
    "LLM_PROVIDERS",
    "ConcurrencyLimitedLLM",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "Message",
    "create_llm",
]
