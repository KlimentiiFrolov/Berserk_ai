"""Отправка в Telegram с ограниченным ожиданием и без слепых повторов."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.types import CallbackQuery, Message

from core.config import get_settings
from core.logger import prepare_logger

logger = prepare_logger(__name__, get_settings().log)


async def telegram_call(operation: Callable[[], Awaitable[Any]]) -> bool:
    settings = get_settings().bot
    for attempt in range(2):
        try:
            async with asyncio.timeout(settings.telegram_timeout_s):
                await operation()
            return True
        except TelegramRetryAfter as error:
            if attempt == 0 and 0 <= error.retry_after <= settings.telegram_retry_after_s:
                await asyncio.sleep(error.retry_after)
                continue
            logger.warning("Telegram ограничил частоту запросов; отправка прекращена")
            return False
        except (TelegramAPIError, TimeoutError, OSError):
            # При сетевом сбое неизвестно, принял ли Telegram сообщение.
            # Повтор может создать дубль; второе сообщение об ошибке тоже не посылаем.
            logger.exception("Не удалось выполнить запрос к Telegram")
            return False
    return False


async def send_message(message: Message, text: str, **kwargs) -> bool:
    return await telegram_call(lambda: message.answer(text, **kwargs))


async def answer_callback(callback: CallbackQuery, text: str) -> bool:
    return await telegram_call(lambda: callback.answer(text))


@asynccontextmanager
async def typing(bot: Bot, message: Message):
    async def worker():
        while await telegram_call(
            lambda: bot.send_chat_action(
                chat_id=message.chat.id,
                action="typing",
                message_thread_id=message.message_thread_id,
            )
        ):
            await asyncio.sleep(4)

    task = asyncio.create_task(worker())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
