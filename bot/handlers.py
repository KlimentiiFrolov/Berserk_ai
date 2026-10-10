from aiogram import Bot
from aiogram.types import CallbackQuery, Message
from aiogram.utils.chat_action import ChatActionSender

import rag
from bot.formatting import split_answer
from bot.keyboards import ANSWER_FEEDBACK, MAIN_MENU
from core.config import get_settings
from core.logger import prepare_logger
from rag.llm import LLMTimeoutError, LLMUnavailableError

logger = prepare_logger(__name__, get_settings().log)


async def start(message: Message) -> None:
    await message.answer(
        "Привет! Я Берсерк AI.\n\n" + get_settings().bot.help_text,
        parse_mode=None,
        reply_markup=MAIN_MENU,
    )


async def help_command(message: Message) -> None:
    await message.answer(get_settings().bot.help_text, parse_mode=None, reply_markup=MAIN_MENU)


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
    await message.answer(texts[message.text], parse_mode=None, reply_markup=MAIN_MENU)


async def feedback_stub(callback: CallbackQuery) -> None:
    # Только убрать индикатор ожидания Telegram. Оценку пока не сохраняем.
    await callback.answer()


async def unknown_command(message: Message) -> None:
    await message.answer("Неизвестная команда. Список команд: /help", parse_mode=None)


async def question(message: Message, bot: Bot) -> None:
    text = (message.text or "").strip()
    if not text:
        await message.answer("Напиши непустой текстовый вопрос.", parse_mode=None)
        return

    has_answer = False
    async with ChatActionSender.typing(
        bot=bot,
        chat_id=message.chat.id,
        message_thread_id=message.message_thread_id,
    ):
        try:
            result = await rag.answer(text)
            has_answer = bool(result.text.strip())
            response = result.text.strip() or "Не удалось сформировать ответ. Попробуй ещё раз."
        except LLMTimeoutError:
            logger.exception("Превышено время ожидания LLM")
            response = "Ответ занял слишком много времени. Попробуй задать вопрос позже."
        except LLMUnavailableError:
            logger.exception("LLM недоступна")
            response = "Сервис ответов временно недоступен. Попробуй позже."
        except Exception:
            logger.exception("Ошибка обработки вопроса в RAG")
            response = "Не удалось обработать вопрос. Попробуй позже."

    parts = split_answer(response)
    for index, part in enumerate(parts):
        # Для длинного ответа оценка отображается только под последней частью.
        markup = ANSWER_FEEDBACK if has_answer and index == len(parts) - 1 else None
        await message.answer(part, parse_mode=None, reply_markup=markup)


async def unsupported_message(message: Message) -> None:
    await message.answer(
        "Пока я понимаю только текст. Напиши свой вопрос сообщением.", parse_mode=None
    )
