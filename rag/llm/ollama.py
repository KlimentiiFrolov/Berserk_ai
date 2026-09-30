import time

import httpx
import ollama

from core.config import LLMSettings, get_settings
from core.logger import prepare_logger
from rag.llm.base import LLMError, LLMResponse, LLMTimeoutError, LLMUnavailableError, Message

logger = prepare_logger(__name__, get_settings().log)


class OllamaClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        api_key = settings.api_key.get_secret_value()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self.client = ollama.AsyncClient(
            host=settings.base_url,
            timeout=settings.timeout_s,
            headers=headers,
        )
        logger.info(
            "ollama: клиент создан base_url=%s model=%s timeout=%ss",
            settings.base_url, settings.model, settings.timeout_s,
        )

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        options = {
            "temperature": self.settings.temperature if temperature is None else temperature,
            "num_predict": self.settings.max_tokens if max_tokens is None else max_tokens,
        }
        logger.debug(
            "ollama: запрос messages=%d chars=%d options=%s",
            len(messages), sum(len(m.content) for m in messages), options,
        )
        started = time.perf_counter()
        try:
            response = await self.client.chat(
                model=self.settings.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                options=options,
            )
        except httpx.TimeoutException as e:
            logger.warning("ollama: таймаут через %.2f с", time.perf_counter() - started)
            raise LLMTimeoutError(f"Ollama не ответила за {self.settings.timeout_s} с") from e
        except (ConnectionError, httpx.TransportError) as e:
            logger.error("ollama: недоступна base_url=%s: %r", self.settings.base_url, e)
            raise LLMUnavailableError(f"Ollama недоступна: {self.settings.base_url}") from e
        except ollama.ResponseError as e:
            logger.error("ollama: ошибка %s: %s", e.status_code, e.error)
            error_cls = LLMUnavailableError if e.status_code >= 500 else LLMError
            raise error_cls(f"Ollama вернула {e.status_code}: {e.error}") from e

        logger.info(
            "ollama: ответ за %.2f с model=%s prompt_tokens=%s completion_tokens=%s",
            time.perf_counter() - started,
            response.model, response.prompt_eval_count, response.eval_count,
        )
        return LLMResponse(
            text=response.message.content or "",
            model=response.model or self.settings.model,
            prompt_tokens=response.prompt_eval_count,
            completion_tokens=response.eval_count,
        )

    async def close(self) -> None:
        await self.client.close()
        logger.debug("ollama: клиент закрыт")
