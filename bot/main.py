import sqlite3

from aiogram import Bot, Dispatcher
from aiogram.utils.token import TokenValidationError, validate_token

import rag
from bot.routing import create_router, get_bot_commands
from core.config import get_settings
from core.logger import prepare_logger
from storage.feedback import FeedbackStore


async def main() -> None:
    settings = get_settings()
    token = settings.bot.token.get_secret_value()
    try:
        validate_token(token)
    except TokenValidationError:
        raise SystemExit("Укажи корректный BOT__TOKEN в .env (токен от @BotFather).") from None

    prepare_logger("aiogram", settings.log)
    logger = prepare_logger(__name__, settings.log)
    bot = Bot(token=token)
    dispatcher = Dispatcher()
    feedback_store = FeedbackStore(settings.bot.feedback_db_path)
    dispatcher["feedback_store"] = feedback_store
    dispatcher.include_router(create_router())
    try:
        # Проверить доступ к Telegram до загрузки тяжёлых моделей.
        await bot.get_me()
        try:
            await feedback_store.initialize()
        except (OSError, sqlite3.Error):
            logger.exception("Не удалось подготовить хранилище feedback")
        await rag.init(settings)
        await bot.set_my_commands(get_bot_commands())
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
