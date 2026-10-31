import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rag.errors import RetrievalUnavailableError
from rag.retrieval import Retriever


def fake_retriever():
    retriever = Retriever.__new__(Retriever)
    retriever._embed_dense = AsyncMock(return_value=[[1.0]])
    retriever._embed_sparse = AsyncMock(return_value=[None])
    retriever.client = SimpleNamespace(query_points=AsyncMock())
    return retriever


@pytest.mark.parametrize("partial", [False, True])
async def test_qdrant_failure_is_not_an_empty_result(partial):
    retriever = fake_retriever()
    error = ConnectionError("Qdrant unavailable")
    retriever.client.query_points.side_effect = [
        SimpleNamespace(points=[]) if partial else error,
        error,
    ]
    with pytest.raises(RetrievalUnavailableError):
        await retriever._search("Вопрос", ["rules", "faq"])


async def test_empty_search_skips_reranker():
    retriever = fake_retriever()
    retriever.client.query_points.return_value = SimpleNamespace(points=[])
    retriever._rerank = AsyncMock()
    assert await retriever.retrieve("Вопрос", collections=["rules"]) == []
    retriever._rerank.assert_not_awaited()


async def test_search_cancellation_is_not_reported_as_unavailable():
    retriever = fake_retriever()
    retriever.client.query_points.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await retriever._search("Вопрос", ["rules"])
