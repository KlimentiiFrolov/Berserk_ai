import asyncio
import time

from core.config import get_settings
from core.logger import prepare_logger
from rag.llm.base import LLMClient, LLMResponse, Message

logger = prepare_logger(__name__, get_settings().log)


class ConcurrencyLimitedLLM:
    def __init__(self, inner: LLMClient, max_concurrency: int) -> None:
        self.inner = inner
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def is_busy(self) -> bool:
        # True — новый запрос встанет в очередь; бот может предупредить пользователя
        return self._semaphore.locked()

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        queued = self.is_busy
        if queued:
            logger.info("llm: все слоты заняты, запрос ждёт в очереди")
        started = time.perf_counter()
        async with self._semaphore:
            if queued:
                waited = time.perf_counter() - started
                logger.info("llm: запрос вышел из очереди через %.2f с", waited)
            return await self.inner.generate(
                messages, temperature=temperature, max_tokens=max_tokens
            )

    async def close(self) -> None:
        await self.inner.close()
