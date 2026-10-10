import logging

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand
from aiogram.utils.token import TokenValidationError, validate_token

import rag
from bot.handlers import create_router
from core.config import get_settings


async def main() -> None:
    settings = get_settings()
    token = settings.bot.token.get_secret_value()
    try:
        validate_token(token)
    except TokenValidationError:
        raise SystemExit("Укажи корректный BOT__TOKEN в .env (токен от @BotFather).") from None

    logging.basicConfig(
        level=settings.log.level,
        format=settings.log.format,
        datefmt=settings.log.datefmt,
    )
    bot = Bot(token=token)
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    try:
        # Проверить доступ к Telegram до загрузки тяжёлых моделей.
        await bot.get_me()
        await rag.init(settings)
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Начать"),
                BotCommand(command="help", description="Помощь"),
            ]
        )
        await bot.delete_webhook(drop_pending_updates=False)
        await dispatcher.start_polling(
            bot,
            allowed_updates=dispatcher.resolve_used_update_types(),
            close_bot_session=False,
        )
    finally:
        try:
            await rag.close()
        finally:
            await bot.session.close()
