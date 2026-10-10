import asyncio
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import AnswerCallbackQuery, SendChatAction, SendMessage
from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardMarkup, Update

from bot import routing
from bot.routing import create_router
from core.config import BotSettings, Settings, get_settings
from rag.llm import LLMTimeoutError, LLMUnavailableError
from rag.schemas import Answer, AnswerGuardState


@pytest.fixture
def environment(monkeypatch):
    settings = get_settings().model_copy(
        update={
            "bot": BotSettings(token="123456:TEST_TOKEN", developers=[]),
        }
    )
    monkeypatch.setattr("bot.handlers.get_settings", lambda: settings)
    bot = Bot(token="123456:TEST_TOKEN")
    request = AsyncMock(return_value=True)
    monkeypatch.setattr(bot.session, "make_request", request)
    answer = AsyncMock(return_value=Answer("Ответ", AnswerGuardState.OK))
    monkeypatch.setattr("rag.answer", answer)
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    return bot, dispatcher, request, answer


async def test_registry_adds_command_to_routing_and_menu(environment, monkeypatch):
    async def ping(message):
        await message.answer("pong")

    monkeypatch.setattr(
        routing, "COMMANDS", (*routing.COMMANDS, routing.CommandSpec("ping", "Проверка", ping))
    )
    bot, _, request, answer = environment
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    calls = await dispatch((bot, dispatcher, request, answer), "/ping")
    assert any(isinstance(call, SendMessage) and call.text == "pong" for call in calls)
    assert any(
        command.command == "ping" and command.description == "Проверка"
        for command in routing.get_bot_commands()
    )
    answer.assert_not_awaited()


async def dispatch(environment, text=None):
    bot, dispatcher, request, _ = environment
    message = {
        "message_id": 1,
        "date": 0,
        "chat": {"id": 42, "type": "private"},
        "from": {"id": 42, "is_bot": False, "first_name": "Игрок"},
    }
    if text is not None:
        message["text"] = text
        if text.startswith("/"):
            message["entities"] = [
                {"type": "bot_command", "offset": 0, "length": len(text.split()[0])}
            ]
    else:
        message["photo"] = [{"file_id": "photo", "file_unique_id": "p", "width": 1, "height": 1}]
    try:
        await dispatcher.feed_update(
            bot, Update.model_validate({"update_id": 1, "message": message})
        )
    finally:
        await bot.session.close()
    return [call.args[1] for call in request.await_args_list]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("/start", "Привет"),
        ("/help", "Например"),
        ("Справка", "Например"),
        ("Об игре", "коллекционная карточная игра"),
        ("О боте", "Берсерк AI"),
        ("Контакты", "Контакты авторов пока не указаны"),
        ("Настройки", "в разработке"),
        ("/unknown", "Неизвестная команда"),
        (None, "только текст"),
        ("   ", "непустой"),
    ],
)
async def test_non_questions_do_not_call_rag(environment, text, expected):
    calls = await dispatch(environment, text)
    assert any(isinstance(call, SendMessage) and expected in call.text for call in calls)
    environment[3].assert_not_awaited()


async def test_question_typing_and_long_plain_text_answer(environment):
    text = "<b>Ответ</b> 🃏 " * 1000

    async def slow_answer(question):
        # Дать фоновой задаче ChatActionSender отправить typing.
        await asyncio.sleep(0.05)
        return Answer(text, AnswerGuardState.OK)

    environment[3].side_effect = slow_answer
    calls = await dispatch(environment, "  Как работает атака?  ")
    environment[3].assert_awaited_once_with("Как работает атака?")
    typing = [call for call in calls if isinstance(call, SendChatAction)]
    assert typing and typing[0].action == "typing" and typing[0].chat_id == 42
    messages = [call for call in calls if isinstance(call, SendMessage)]
    assert len(messages) > 1
    assert "".join(call.text for call in messages) == text.strip()
    assert all(call.parse_mode is None for call in messages)
    assert all(call.reply_markup is None for call in messages[:-1])
    assert isinstance(messages[-1].reply_markup, InlineKeyboardMarkup)
    assert [button.text for button in messages[-1].reply_markup.inline_keyboard[0]] == ["👍", "👎"]


@pytest.mark.parametrize(
    "error, expected",
    [
        (LLMTimeoutError("internal"), "слишком много времени"),
        (LLMUnavailableError("internal"), "временно недоступен"),
        (RuntimeError("internal"), "Не удалось обработать"),
    ],
)
async def test_rag_errors_are_reported_without_internal_details(environment, error, expected):
    environment[3].side_effect = error
    calls = await dispatch(environment, "Вопрос")
    messages = [call.text for call in calls if isinstance(call, SendMessage)]
    assert len(messages) == 1
    assert expected in messages[0]
    assert "internal" not in messages[0]
    assert all(call.reply_markup is None for call in calls if isinstance(call, SendMessage))


async def test_start_shows_reply_keyboard(environment):
    calls = await dispatch(environment, "/start")
    keyboard = next(call.reply_markup for call in calls if isinstance(call, SendMessage))
    assert isinstance(keyboard, ReplyKeyboardMarkup)
    assert {button.text for row in keyboard.keyboard for button in row} == {
        "Справка",
        "Об игре",
        "О боте",
        "Контакты",
        "Настройки",
    }


async def test_contacts_from_env(environment, monkeypatch):
    defaults = get_settings()
    monkeypatch.setenv("BOT__TOKEN", "123456:TEST_TOKEN")
    monkeypatch.setenv("BOT__DEVELOPERS", '["Автор — @example", "  ", "Другой — @example2"]')
    settings = Settings(_env_file=None, qdrant=defaults.qdrant, llm=defaults.llm)
    monkeypatch.setattr("bot.handlers.get_settings", lambda: settings)
    calls = await dispatch(environment, "Контакты")
    messages = [call for call in calls if isinstance(call, SendMessage)]
    assert messages[0].text == "Контакты авторов бота:\n\nАвтор — @example\nДругой — @example2"
    environment[3].assert_not_awaited()


@pytest.mark.parametrize("data", ["feedback:like", "feedback:dislike"])
async def test_feedback_only_acknowledges_callback(environment, data):
    bot, dispatcher, request, answer = environment
    update = Update.model_validate(
        {
            "update_id": 2,
            "callback_query": {
                "id": "feedback-query",
                "chat_instance": "chat",
                "from": {"id": 42, "is_bot": False, "first_name": "Игрок"},
                "data": data,
            },
        }
    )
    try:
        await dispatcher.feed_update(bot, update)
    finally:
        await bot.session.close()
    answer.assert_not_awaited()
    request.assert_awaited_once()
    method = request.await_args.args[1]
    assert isinstance(method, AnswerCallbackQuery)
    assert method.callback_query_id == "feedback-query"
    assert method.text is None


@pytest.mark.parametrize(
    "text, state, expected",
    [
        ("Нет информации", AnswerGuardState.REFUSE, "Нет информации"),
        ("  ", AnswerGuardState.OK, "Не удалось сформировать ответ"),
    ],
)
async def test_refusal_and_empty_answer(environment, text, state, expected):
    environment[3].return_value = Answer(text, state)
    calls = await dispatch(environment, "Вопрос")
    assert any(isinstance(call, SendMessage) and expected in call.text for call in calls)
