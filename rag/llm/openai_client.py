import time

import openai

from core.config import LLMSettings, get_settings
from core.logger import prepare_logger
from rag.llm.base import LLMError, LLMResponse, LLMTimeoutError, LLMUnavailableError, Message

logger = prepare_logger(__name__, get_settings().log)


class OpenAIClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.client = openai.AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.timeout_s,
            max_retries=0,  # повторы — забота вызывающего кода, не адаптера
        )
        logger.info(
            "openai: клиент создан base_url=%s model=%s timeout=%ss",
            settings.base_url, settings.model, settings.timeout_s,
        )

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        temperature = self.settings.temperature if temperature is None else temperature
        max_tokens = self.settings.max_tokens if max_tokens is None else max_tokens
        logger.debug(
            "openai: запрос messages=%d chars=%d temperature=%s max_tokens=%s",
            len(messages), sum(len(m.content) for m in messages), temperature, max_tokens,
        )
        started = time.perf_counter()
        try:
            completion = await self.client.chat.completions.create(
                model=self.settings.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.APITimeoutError as e:  # подкласс APIConnectionError — проверяется первым
            logger.warning("openai: таймаут через %.2f с", time.perf_counter() - started)
            raise LLMTimeoutError(f"LLM не ответила за {self.settings.timeout_s} с") from e
        except openai.APIConnectionError as e:
            logger.error("openai: недоступна base_url=%s: %r", self.settings.base_url, e)
            raise LLMUnavailableError(f"LLM недоступна: {self.settings.base_url}") from e
        except openai.APIStatusError as e:
            logger.error("openai: ошибка %s: %s", e.status_code, e.message)
            error_cls = LLMUnavailableError if e.status_code >= 500 else LLMError
            raise error_cls(f"LLM вернула {e.status_code}: {e.message}") from e

        usage = completion.usage
        logger.info(
            "openai: ответ за %.2f с model=%s prompt_tokens=%s completion_tokens=%s",
            time.perf_counter() - started,
            completion.model,
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
        )
        return LLMResponse(
            text=completion.choices[0].message.content or "",
            model=completion.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )

    async def close(self) -> None:
        await self.client.close()
        logger.debug("openai: клиент закрыт")
