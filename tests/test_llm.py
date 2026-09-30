import asyncio
import json
from types import SimpleNamespace

import httpx
import httpx2
import openai
import pytest

from core.config import LLMSettings
from rag.llm import (
    ConcurrencyLimitedLLM,
    LLMError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
    create_llm,
)
from rag.llm.ollama import OllamaClient
from rag.llm.openai_client import OpenAIClient

MESSAGES = [Message("system", "Ты помощник"), Message("user", "Что такое атака?")]


def ollama_settings(**kw) -> LLMSettings:
    return LLMSettings(
        provider="ollama", base_url="http://ollama.test", model="qwen", api_key="", **kw
    )


def openai_settings(**kw) -> LLMSettings:
    return LLMSettings(
        provider="openai", base_url="http://openai.test/v1", model="gpt", api_key="sk-test", **kw
    )


# --- Ollama ---

def make_ollama(handler, settings: LLMSettings | None = None) -> OllamaClient:
    settings = settings or ollama_settings()
    client = OllamaClient(settings)
    # подменяем httpx-клиент внутри SDK: проверяется и SDK, и перевод ошибок адаптером
    client.client._client = httpx.AsyncClient(
        base_url=settings.base_url, transport=httpx.MockTransport(handler)
    )
    return client


async def test_ollama_request_and_response_mapping():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "qwen",
            "message": {"role": "assistant", "content": "Атака — это..."},
            "prompt_eval_count": 12,
            "eval_count": 5,
        })

    client = make_ollama(handler, ollama_settings(temperature=0.3, max_tokens=100))
    response = await client.generate(MESSAGES, max_tokens=50)

    assert response == LLMResponse("Атака — это...", "qwen", 12, 5)
    assert seen["path"] == "/api/chat"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"][1] == {"role": "user", "content": "Что такое атака?"}
    assert seen["body"]["options"] == {"temperature": 0.3, "num_predict": 50}  # аргумент > конфиг


@pytest.mark.parametrize(
    ("handler_error", "expected"),
    [
        (httpx.ReadTimeout("timeout"), LLMTimeoutError),
        (httpx.ConnectError("refused"), LLMUnavailableError),
    ],
)
async def test_ollama_transport_errors(handler_error, expected):
    def handler(request):
        raise handler_error

    with pytest.raises(expected):
        await make_ollama(handler).generate(MESSAGES)


@pytest.mark.parametrize(("status", "expected"), [(404, LLMError), (503, LLMUnavailableError)])
async def test_ollama_status_errors(status, expected):
    client = make_ollama(lambda request: httpx.Response(status, text="model not found"))
    with pytest.raises(expected) as exc_info:
        await client.generate(MESSAGES)
    assert type(exc_info.value) is expected


# --- OpenAI-совместимый ---

def fake_completion(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
    )


async def test_openai_request_and_response_mapping(monkeypatch):
    client = OpenAIClient(openai_settings(temperature=0.1, max_tokens=200))
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return fake_completion("Ответ")

    monkeypatch.setattr(client.client.chat.completions, "create", create)
    response = await client.generate(MESSAGES, temperature=0.7)

    assert response == LLMResponse("Ответ", "gpt", 10, 3)
    assert seen["model"] == "gpt"
    assert seen["temperature"] == 0.7
    assert seen["max_tokens"] == 200
    assert seen["messages"][0] == {"role": "system", "content": "Ты помощник"}


REQUEST = httpx2.Request("POST", "http://openai.test/v1/chat/completions")


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (openai.APITimeoutError(request=REQUEST), LLMTimeoutError),
        (openai.APIConnectionError(request=REQUEST), LLMUnavailableError),
        (
            openai.APIStatusError(
                "bad", response=httpx2.Response(401, request=REQUEST), body=None
            ),
            LLMError,
        ),
        (
            openai.APIStatusError(
                "down", response=httpx2.Response(502, request=REQUEST), body=None
            ),
            LLMUnavailableError,
        ),
    ],
)
async def test_openai_errors(monkeypatch, sdk_error, expected):
    client = OpenAIClient(openai_settings())

    async def create(**kwargs):
        raise sdk_error

    monkeypatch.setattr(client.client.chat.completions, "create", create)
    with pytest.raises(expected) as exc_info:
        await client.generate(MESSAGES)
    assert type(exc_info.value) is expected


# --- Очередь и фабрика ---

class SlowLLM:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def generate(self, messages, *, temperature=None, max_tokens=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.05)
        self.active -= 1
        return LLMResponse("ok", "fake")

    async def close(self):
        pass


async def test_limiter_serializes_requests():
    inner = SlowLLM()
    llm = ConcurrencyLimitedLLM(inner, max_concurrency=1)

    assert not llm.is_busy
    first = asyncio.create_task(llm.generate(MESSAGES))
    await asyncio.sleep(0.01)
    assert llm.is_busy

    await asyncio.gather(first, llm.generate(MESSAGES), llm.generate(MESSAGES))
    assert inner.max_active == 1
    assert not llm.is_busy


async def test_limiter_allows_parallel_requests():
    inner = SlowLLM()
    llm = ConcurrencyLimitedLLM(inner, max_concurrency=3)
    await asyncio.gather(*(llm.generate(MESSAGES) for _ in range(3)))
    assert inner.max_active == 3


@pytest.mark.parametrize(
    ("settings", "client_cls"),
    [(ollama_settings(), OllamaClient), (openai_settings(), OpenAIClient)],
)
async def test_create_llm_picks_provider(settings, client_cls):
    llm = create_llm(settings)
    assert isinstance(llm, ConcurrencyLimitedLLM)
    assert isinstance(llm.inner, client_cls)
    await llm.close()
