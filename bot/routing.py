"""Реестр маршрутов: специальные обработчики идут перед общими.

Новую команду добавляют в COMMANDS: запись определяет и маршрутизацию,
и меню команд Telegram. Обработчики кнопок добавляют в MESSAGE_HANDLERS,
а обработчики callback - в CALLBACK_HANDLERS. Порядок записей значим.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand

from bot import handlers

Handler = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    handler: Handler
    command_filter: Command | None = None


@dataclass(frozen=True)
class HandlerSpec:
    handler: Handler
    filters: tuple[Any, ...] = ()


COMMANDS = (
    CommandSpec("start", "Начать", handlers.start, CommandStart()),
    CommandSpec("help", "Помощь", handlers.help_command),
)

MESSAGE_HANDLERS = (
    HandlerSpec(handlers.help_command, (F.text == "Справка",)),
    HandlerSpec(handlers.menu_info, (F.text.in_({"Об игре", "О боте", "Контакты", "Настройки"}),)),
)

# Эти обработчики всегда регистрируются после команд и кнопок меню.
FALLBACK_HANDLERS = (
    HandlerSpec(handlers.unknown_command, (F.text.startswith("/"),)),
    HandlerSpec(handlers.question, (F.text,)),
    HandlerSpec(handlers.unsupported_message),
)

CALLBACK_HANDLERS = (
    HandlerSpec(handlers.feedback_stub, (F.data.in_({"feedback:like", "feedback:dislike"}),)),
)


def get_bot_commands() -> list[BotCommand]:
    return [BotCommand(command=entry.name, description=entry.description) for entry in COMMANDS]


def create_router() -> Router:
    router = Router(name="bot")
    for command in COMMANDS:
        router.message.register(command.handler, command.command_filter or Command(command.name))
    for entry in (*MESSAGE_HANDLERS, *FALLBACK_HANDLERS):
        router.message.register(entry.handler, *entry.filters)
    for entry in CALLBACK_HANDLERS:
        router.callback_query.register(entry.handler, *entry.filters)
    return router
