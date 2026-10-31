import asyncio
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)
from aiogram.methods import SendMessage

from bot import transport
from core.config import BotSettings, get_settings

METHOD = SendMessage(chat_id=42, text="Ответ")


@pytest.fixture
def settings(monkeypatch):
    settings = get_settings().model_copy(
        update={
            "bot": BotSettings(token="123456:TEST", telegram_timeout_s=0.02),
        }
    )
    monkeypatch.setattr(transport, "get_settings", lambda: settings)
    return settings


@pytest.mark.parametrize(
    "error_type",
    [
        TelegramBadRequest,
        TelegramForbiddenError,
        TelegramNetworkError,
        TelegramServerError,
    ],
)
async def test_delivery_failure_is_not_retried(settings, error_type):
    operation = AsyncMock(side_effect=error_type(method=METHOD, message="failed"))
    assert not await transport.telegram_call(operation)
    operation.assert_awaited_once()


async def test_retry_after_retries_once(settings, monkeypatch):
    error = TelegramRetryAfter(method=METHOD, message="limit", retry_after=2)
    operation = AsyncMock(side_effect=[error, True])
    sleep = AsyncMock()
    monkeypatch.setattr(transport.asyncio, "sleep", sleep)
    assert await transport.telegram_call(operation)
    assert operation.await_count == 2
    sleep.assert_awaited_once_with(2)


async def test_repeated_rate_limit_is_bounded(settings):
    operation = AsyncMock(
        side_effect=TelegramRetryAfter(
            method=METHOD,
            message="limit",
            retry_after=0,
        )
    )
    assert not await transport.telegram_call(operation)
    assert operation.await_count == 2


async def test_long_rate_limit_is_not_retried(settings):
    operation = AsyncMock(
        side_effect=TelegramRetryAfter(
            method=METHOD,
            message="limit",
            retry_after=60,
        )
    )
    assert not await transport.telegram_call(operation)
    operation.assert_awaited_once()


async def test_delivery_timeout_cancels_request(settings):
    cancelled = asyncio.Event()

    async def hanging():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    assert not await transport.telegram_call(hanging)
    assert cancelled.is_set()


async def test_external_cancellation_propagates(settings):
    operation = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await transport.telegram_call(operation)
