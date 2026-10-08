"""Единый интерфейс RAG: await rag.answer(question) -> Answer.

rag.init() — один раз при старте (загрузка моделей, подключения), rag.close() — при остановке.
"""
import asyncio

from core.config import Settings, get_settings
from core.logger import prepare_logger
from rag.llm import LLMClient, Message, create_llm
from rag.schemas import Answer, AnswerGuardState

logger = prepare_logger(__name__, get_settings().log)

REFUSAL_TEXT = "Не удалось найти в базе знаний информацию по вашему вопросу."

_settings: Settings | None = None
_retriever = None
_llm: LLMClient | None = None

# TODO: вызывается ОДИН раз при старте бота
async def init(settings: Settings | None = None) -> None:
    global _settings, _retriever, _llm
    _settings = settings or get_settings()

    from rag.retrieval import Retriever  # fastembed тяжёлый

    _retriever = await asyncio.to_thread(Retriever, _settings)  # загрузка моделей блокирующая
    _llm = create_llm(_settings.llm)


async def answer(question: str, history: list[Message] | None = None) -> Answer:
    if _settings is None or _retriever is None or _llm is None:
        raise RuntimeError("RAG не инициализирован: вызовите await rag.init() при старте")

    rs = _settings.retrieval
    chunks = await _retriever.retrieve(
        question,
        top_k=rs.top_k,
        dense_chunks_limit=rs.dense_limit,
        sparse_chunks_limit=rs.sparse_limit,
        rrf_limit=rs.rrf_limit,
        use_reranker=rs.use_reranker,
    )
    collections = sorted({c.chunk.collection for c in chunks})
    logger.info("rag: найдено чанков=%d collections=%s", len(chunks), collections)

    # TODO answer_guard: пороги OK / WARN / REFUSE и источники
    if not chunks:
        return Answer(REFUSAL_TEXT, AnswerGuardState.REFUSE)

    # TODO context_builder: промпт из чанков
    context = "\n\n".join(f"[{i}] {c.chunk.text}" for i, c in enumerate(chunks, start=1))
    messages = [
        Message("system", _settings.llm.system_prompt),
        *(history or []),
        Message("user", f"Контекст:\n{context}\n\nВопрос: {question}"),
    ]
    response = await _llm.generate(messages)

    return Answer(
        text=response.text,
        guard_state=AnswerGuardState.OK,
        routed_collections=collections,
        chunks=chunks,
    )


async def close() -> None:
    global _settings, _retriever, _llm
    if _llm is not None:
        await _llm.close()
    if _retriever is not None:
        await _retriever.close()
    _settings = _retriever = _llm = None
