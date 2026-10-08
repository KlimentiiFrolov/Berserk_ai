import pytest

import rag
from core.config import get_settings
from rag.llm import LLMResponse, Message
from rag.schemas import AnswerGuardState, Chunk, RetrievedChunk


def make_chunk(text: str, collection: str = "rules") -> RetrievedChunk:
    chunk = Chunk(id=text, text=text, source_type="official", collection=collection)
    return RetrievedChunk(chunk=chunk, score=0.9, stage="reranked")


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    async def retrieve(self, query, **kwargs):
        return self.chunks

    async def close(self):
        pass


class FakeLLM:
    def __init__(self) -> None:
        self.messages: list[Message] | None = None

    async def generate(self, messages, *, temperature=None, max_tokens=None):
        self.messages = messages
        return LLMResponse("ответ LLM", "fake")

    async def close(self):
        pass


@pytest.fixture
def fake_rag(monkeypatch):
    """Подменяет зависимости rag без Qdrant, моделей и LLM."""
    def setup(chunks):
        llm = FakeLLM()
        monkeypatch.setattr(rag, "_settings", get_settings())
        monkeypatch.setattr(rag, "_retriever", FakeRetriever(chunks))
        monkeypatch.setattr(rag, "_llm", llm)
        return llm
    return setup


async def test_answer(fake_rag):
    llm = fake_rag([make_chunk("правило об атаке", "rules"), make_chunk("карта", "cards")])

    answer = await rag.answer("Что такое атака?")

    assert answer.text == "ответ LLM"
    assert answer.guard_state is AnswerGuardState.OK
    assert answer.routed_collections == ["cards", "rules"]
    assert "правило об атаке" in llm.messages[-1].content
    assert "Что такое атака?" in llm.messages[-1].content


async def test_no_chunks_refuses_without_llm(fake_rag):
    llm = fake_rag([])
    answer = await rag.answer("вопрос")
    assert answer.guard_state is AnswerGuardState.REFUSE
    assert answer.text == rag.REFUSAL_TEXT
    assert llm.messages is None


async def test_history(fake_rag):
    llm = fake_rag([make_chunk("текст")])
    await rag.answer("новый", [Message("user", "старый"), Message("assistant", "ответ")])
    assert [m.role for m in llm.messages] == ["system", "user", "assistant", "user"]


async def test_answer_requires_init():
    with pytest.raises(RuntimeError, match="rag.init"):
        await rag.answer("вопрос")
