import asyncio
import sqlite3
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import AnswerCallbackQuery, SendChatAction, SendMessage
from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardMarkup, Update

from bot import handlers, routing
from bot.routing import create_router
from core.config import BotSettings, Settings, get_settings
from rag.errors import RetrievalUnavailableError
from rag.llm import LLMTimeoutError, LLMUnavailableError
from rag.schemas import Answer, AnswerGuardState, Source
from storage.feedback import FeedbackStore


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
    store = AsyncMock()
    store.save_answer.return_value = 42
    store.save_rating.return_value = True
    dispatcher["feedback_store"] = store
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
    assert messages[-1].reply_markup.inline_keyboard[0][0].callback_data == "feedback:42:like"
    environment[1]["feedback_store"].save_answer.assert_awaited_once_with(
        "Как работает атака?",
        text.strip(),
        42,
        sources=[],
    )


@pytest.mark.parametrize(
    "error, expected",
    [
        (LLMTimeoutError("internal"), "слишком много времени"),
        (LLMUnavailableError("internal"), "временно недоступен"),
        (RetrievalUnavailableError("internal"), "База знаний временно недоступна"),
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
async def test_legacy_feedback_is_unavailable(environment, data):
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
    assert method.text == "Этот ответ больше недоступен."


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


async def send_feedback(environment, data):
    bot, dispatcher, request, _ = environment
    request.reset_mock()
    update = Update.model_validate(
        {
            "update_id": 3,
            "callback_query": {
                "id": "vote",
                "chat_instance": "chat",
                "data": data,
                "from": {"id": 7, "is_bot": False, "first_name": "Игрок"},
                "message": {"message_id": 2, "date": 1, "chat": {"id": 42, "type": "private"}},
            },
        }
    )
    await dispatcher.feed_update(bot, update)
    return request.await_args.args[1]


@pytest.mark.parametrize("action, rating", [("like", 1), ("dislike", -1)])
async def test_feedback_saves_user_and_answer(environment, action, rating):
    result = await send_feedback(environment, f"feedback:42:{action}")
    environment[1]["feedback_store"].save_rating.assert_awaited_once_with(
        answer_id=42,
        user_id=7,
        rating=rating,
        chat_id=42,
    )
    assert isinstance(result, AnswerCallbackQuery)
    assert result.text == "Оценка сохранена."


@pytest.mark.parametrize(
    "data",
    ["feedback:like", "feedback:0:like", "feedback:42:no", "feedback:9999999999999999999:like"],
)
async def test_invalid_feedback_does_not_write(environment, data):
    result = await send_feedback(environment, data)
    environment[1]["feedback_store"].save_rating.assert_not_awaited()
    assert result.text == "Этот ответ больше недоступен."


async def test_unavailable_answer(environment):
    environment[1]["feedback_store"].save_rating.return_value = False
    result = await send_feedback(environment, "feedback:42:like")
    assert result.text == "Этот ответ больше недоступен."


async def test_feedback_database_failure(environment):
    environment[1]["feedback_store"].save_rating.side_effect = sqlite3.OperationalError("locked")
    result = await send_feedback(environment, "feedback:42:like")
    assert "Не удалось сохранить оценку" in result.text


async def test_answer_database_failure_still_sends_text(environment):
    environment[1]["feedback_store"].save_answer.side_effect = sqlite3.OperationalError("locked")
    calls = await dispatch(environment, "Вопрос")
    messages = [call for call in calls if isinstance(call, SendMessage)]
    assert messages[0].text == "Ответ"
    assert messages[0].reply_markup is None


async def test_question_and_feedback_with_real_database(environment, tmp_path):
    import aiosqlite

    path = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(path)
    await store.initialize()
    environment[1]["feedback_store"] = store
    calls = await dispatch(environment, "Как работает атака?")
    message = next(call for call in calls if isinstance(call, SendMessage))
    buttons = message.reply_markup.inline_keyboard[0]
    # После переоткрытия БД кнопки уже отправленного сообщения продолжают работать.
    environment[1]["feedback_store"] = FeedbackStore(path)
    await send_feedback(environment, buttons[0].callback_data)
    await send_feedback(environment, buttons[1].callback_data)
    async with aiosqlite.connect(path) as db:
        async with db.execute(
            "SELECT a.question, a.answer, f.user_id, f.rating "
            "FROM answers a JOIN feedback f ON f.answer_id = a.id"
        ) as cursor:
            assert await cursor.fetchall() == [("Как работает атака?", "Ответ", 7, -1)]


async def test_sources_sent_split_and_stored_with_feedback(environment):
    sources = [
        Source("rules", "official", "Правила: 3.2", "https://example.org/rules"),
        Source("community", "community", "Обсуждение: атака"),
    ]
    body = "Текст ответа. " * 400
    environment[3].return_value = Answer(body, AnswerGuardState.OK, sources=sources)
    calls = await dispatch(environment, "Вопрос")
    messages = [call for call in calls if isinstance(call, SendMessage)]
    combined = "".join(call.text for call in messages)
    assert combined.startswith(body.strip() + "\n\nИсточники:")
    assert "Правила: 3.2 (официальный источник)" in combined
    assert "https://example.org/rules" in combined
    assert "Обсуждение: атака (сообщество)" in combined
    assert all(len(call.text.encode("utf-16-le")) // 2 <= 4096 for call in messages)
    assert all(call.reply_markup is None for call in messages[:-1])
    assert isinstance(messages[-1].reply_markup, InlineKeyboardMarkup)
    assert all(
        call.parse_mode is None and call.link_preview_options.is_disabled for call in messages
    )
    environment[1]["feedback_store"].save_answer.assert_awaited_once_with(
        "Вопрос",
        body.strip(),
        42,
        sources=sources,
    )


async def test_total_timeout_cancels_rag_and_next_question_works(environment):
    handlers.get_settings().bot.request_timeout_s = 0.03
    cancelled = asyncio.Event()

    async def hanging(question):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    environment[3].side_effect = hanging
    calls = await dispatch(environment, "Медленный вопрос")
    assert cancelled.is_set()
    assert any(
        isinstance(call, SendMessage) and "слишком много времени" in call.text for call in calls
    )
    environment[1]["feedback_store"].save_answer.assert_not_awaited()
    environment[3].side_effect = None
    environment[2].reset_mock()
    calls = await dispatch(environment, "Следующий вопрос")
    assert any(isinstance(call, SendMessage) and call.text == "Ответ" for call in calls)


async def test_typing_network_failure_does_not_lose_answer(environment):
    async def request(bot, method, **kwargs):
        if isinstance(method, SendChatAction):
            raise TelegramNetworkError(method=method, message="typing failed")
        return True

    async def slow(question):
        await asyncio.sleep(0.03)
        return Answer("Ответ", AnswerGuardState.OK)

    environment[2].side_effect = request
    environment[3].side_effect = slow
    calls = await dispatch(environment, "Вопрос")
    assert any(isinstance(call, SendChatAction) for call in calls)
    assert any(isinstance(call, SendMessage) and call.text == "Ответ" for call in calls)


async def test_partial_delivery_stops_without_resending_previous_parts(environment):
    sent = []

    async def request(bot, method, **kwargs):
        if isinstance(method, SendMessage):
            sent.append(method.text)
            if len(sent) == 2:
                raise TelegramNetworkError(method=method, message="connection lost")
        return True

    environment[2].side_effect = request
    environment[3].return_value = Answer("а" * 10000, AnswerGuardState.OK)
    await dispatch(environment, "Вопрос")
    assert len(sent) == 2
    environment[3].return_value = Answer("Следующий ответ", AnswerGuardState.OK)
    await dispatch(environment, "Следующий вопрос")
    assert sent[-1] == "Следующий ответ"


async def test_expired_callback_does_not_repeat_database_write(environment):
    environment[2].side_effect = lambda bot, method, **kwargs: expired(method)

    def expired(method):
        raise TelegramBadRequest(method=method, message="query is too old")

    await send_feedback(environment, "feedback:42:like")
    environment[1]["feedback_store"].save_rating.assert_awaited_once()
    environment[2].assert_awaited_once()


async def test_shutdown_cancellation_stops_typing_without_error_message(environment):
    started = asyncio.Event()

    async def hanging(question):
        started.set()
        await asyncio.Event().wait()

    environment[3].side_effect = hanging
    task = asyncio.create_task(dispatch(environment, "Вопрос"))
    await started.wait()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    count = environment[2].await_count
    await asyncio.sleep(0.02)
    assert environment[2].await_count == count
    assert not any(isinstance(call.args[1], SendMessage) for call in environment[2].await_args_list)
