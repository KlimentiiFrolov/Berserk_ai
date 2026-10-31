import asyncio
import re
import sqlite3

from aiogram import Bot
from aiogram.types import CallbackQuery, LinkPreviewOptions, Message

import rag
from bot.formatting import format_sources, split_answer
from bot.keyboards import MAIN_MENU, answer_feedback_keyboard
from bot.transport import answer_callback, send_message, typing
from core.config import get_settings
from core.logger import prepare_logger
from rag.errors import RetrievalUnavailableError
from rag.llm import LLMTimeoutError, LLMUnavailableError
from storage.feedback import FeedbackStore

logger = prepare_logger(__name__, get_settings().log)


async def start(message: Message) -> None:
    await send_message(
        message,
        "Привет! Я Берсерк AI.\n\n" + get_settings().bot.help_text,
        parse_mode=None,
        reply_markup=MAIN_MENU,
    )


async def help_command(message: Message) -> None:
    await send_message(
        message, get_settings().bot.help_text, parse_mode=None, reply_markup=MAIN_MENU
    )


async def menu_info(message: Message) -> None:
    settings = get_settings().bot
    developers = [entry.strip() for entry in settings.developers if entry.strip()]
    contacts = (
        "Контакты авторов бота:\n\n" + "\n".join(developers)
        if developers
        else "Контакты авторов пока не указаны."
    )
    texts = {
        "Об игре": settings.about_game_text,
        "О боте": settings.about_bot_text,
        "Контакты": contacts,
        "Настройки": settings.settings_stub_text,
    }
    await send_message(message, texts[message.text], parse_mode=None, reply_markup=MAIN_MENU)


async def feedback(callback: CallbackQuery, feedback_store: FeedbackStore) -> None:
    match = re.fullmatch(r"feedback:([1-9][0-9]{0,18}):(like|dislike)", callback.data or "")
    if not match or callback.message is None or int(match[1]) > 2**63 - 1:
        await answer_callback(callback, "Этот ответ больше недоступен.")
        return
    try:
        saved = await feedback_store.save_rating(
            answer_id=int(match[1]),
            user_id=callback.from_user.id,
            rating=1 if match[2] == "like" else -1,
            chat_id=callback.message.chat.id,
        )
    except (sqlite3.Error, OSError):
        logger.exception("Не удалось сохранить оценку")
        await answer_callback(callback, "Не удалось сохранить оценку. Попробуй позже.")
        return
    await answer_callback(
        callback, "Оценка сохранена." if saved else "Этот ответ больше недоступен."
    )


async def unknown_command(message: Message) -> None:
    await send_message(message, "Неизвестная команда. Список команд: /help", parse_mode=None)


async def question(message: Message, bot: Bot, feedback_store: FeedbackStore) -> None:
    text = (message.text or "").strip()
    if not text:
        await send_message(message, "Напиши непустой текстовый вопрос.", parse_mode=None)
        return

    has_answer = False
    sources = []
    async with typing(bot, message):
        try:
            async with asyncio.timeout(get_settings().bot.request_timeout_s):
                result = await rag.answer(text)
            has_answer = bool(result.text.strip())
            response = result.text.strip() or "Не удалось сформировать ответ. Попробуй ещё раз."
            if has_answer:
                sources = result.sources
        except (LLMTimeoutError, TimeoutError):
            logger.exception("Превышено время обработки вопроса или ожидания LLM")
            response = "Ответ занял слишком много времени. Попробуй задать вопрос позже."
        except RetrievalUnavailableError:
            logger.exception("Поиск по базе знаний недоступен")
            response = "База знаний временно недоступна. Попробуй позже."
        except LLMUnavailableError:
            logger.exception("LLM недоступна")
            response = "Сервис ответов временно недоступен. Попробуй позже."
        except Exception:
            logger.exception("Ошибка обработки вопроса в RAG")
            response = "Не удалось обработать вопрос. Попробуй позже."

    answer_id = None
    if has_answer:
        try:
            answer_id = await feedback_store.save_answer(
                text, response, message.chat.id, sources=sources
            )
        except (sqlite3.Error, OSError):
            logger.exception("Не удалось сохранить ответ; отправка без кнопок оценки")

    parts = split_answer(response + format_sources(sources))
    for index, part in enumerate(parts):
        # Для длинного ответа оценка отображается только под последней частью.
        markup = (
            answer_feedback_keyboard(answer_id)
            if answer_id is not None and index == len(parts) - 1
            else None
        )
        delivered = await send_message(
            message,
            part,
            parse_mode=None,
            reply_markup=markup,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )
        if not delivered:
            break


async def unsupported_message(message: Message) -> None:
    await send_message(
        message, "Пока я понимаю только текст. Напиши свой вопрос сообщением.", parse_mode=None
    )
