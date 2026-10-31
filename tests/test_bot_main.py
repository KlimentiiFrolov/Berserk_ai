import sqlite3
from unittest.mock import AsyncMock, Mock

import pytest

from bot import main as entrypoint
from core.config import BotSettings, get_settings


@pytest.fixture
def lifecycle(monkeypatch):
    settings = get_settings().model_copy(
        update={
            "bot": BotSettings(token="123456:TEST_TOKEN"),
        }
    )
    monkeypatch.setattr(entrypoint, "get_settings", lambda: settings)
    bot = AsyncMock()
    dispatcher = Mock()
    dispatcher.__setitem__ = Mock()
    store = AsyncMock()
    monkeypatch.setattr(entrypoint, "FeedbackStore", Mock(return_value=store))
    dispatcher.start_polling = AsyncMock()
    dispatcher.resolve_used_update_types.return_value = ["message"]
    monkeypatch.setattr(entrypoint, "Bot", Mock(return_value=bot))
    monkeypatch.setattr(entrypoint, "Dispatcher", Mock(return_value=dispatcher))
    init, close = AsyncMock(), AsyncMock()
    monkeypatch.setattr(entrypoint.rag, "init", init)
    monkeypatch.setattr(entrypoint.rag, "close", close)
    return settings, bot, dispatcher, init, close


async def test_polling_initializes_and_closes_resources(lifecycle):
    settings, bot, dispatcher, init, close = lifecycle
    await entrypoint.main()
    entrypoint.FeedbackStore.assert_called_once_with(settings.bot.feedback_db_path)
    store = entrypoint.FeedbackStore.return_value
    store.initialize.assert_awaited_once()
    dispatcher.__setitem__.assert_called_once_with("feedback_store", store)
    init.assert_awaited_once_with(settings)
    bot.delete_webhook.assert_awaited_once_with(drop_pending_updates=False)
    dispatcher.start_polling.assert_awaited_once_with(
        bot,
        allowed_updates=["message"],
        close_bot_session=False,
    )
    close.assert_awaited_once()
    bot.session.close.assert_awaited_once()


@pytest.mark.parametrize("stage", ["init", "polling", "close"])
async def test_cleanup_on_failure(lifecycle, stage):
    _, bot, dispatcher, init, close = lifecycle
    target = {"init": init, "polling": dispatcher.start_polling, "close": close}[stage]
    target.side_effect = RuntimeError("test failure")
    with pytest.raises(RuntimeError, match="test failure"):
        await entrypoint.main()
    close.assert_awaited_once()
    bot.session.close.assert_awaited_once()


async def test_missing_token_fails_before_initialization(lifecycle):
    settings, _, _, init, _ = lifecycle
    settings.bot = BotSettings(token="")
    with pytest.raises(SystemExit, match="BOT__TOKEN"):
        await entrypoint.main()
    init.assert_not_awaited()


async def test_storage_failure_does_not_prevent_polling(lifecycle):
    _, bot, dispatcher, _, _ = lifecycle
    entrypoint.FeedbackStore.return_value.initialize.side_effect = sqlite3.OperationalError(
        "unable to open database file"
    )
    await entrypoint.main()
    dispatcher.start_polling.assert_awaited_once()
    bot.session.close.assert_awaited_once()
