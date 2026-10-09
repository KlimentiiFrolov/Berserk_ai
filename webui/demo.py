import asyncio
import gradio as gr

import rag
from rag.schemas import AnswerGuardState
from .formatting import _to_text, _history_to_messages, _format_chunks, _format_sources

_init_lock = asyncio.Lock()
_ready = False

_GUARD_PREFIX = {
    AnswerGuardState.OK: "",
    AnswerGuardState.WARN: "*Контекст слабый, ответ может быть неточным.*\n\n",
    AnswerGuardState.REFUSE: "",
}

# AsyncQdrantClient в RAG привязывается к event loop, а gradio запускает свой
async def _ensure_init() -> None:
    global _ready
    if _ready:
        return
    async with _init_lock:
        if not _ready:
            await rag.init()
            _ready = True


async def respond(message, history):
    try:
        await _ensure_init()
        result = await rag.answer(_to_text(message), _history_to_messages(history))
    except Exception as e:
        return f"Ошибка: {type(e).__name__}: {e}"

    header = f"`{result.guard_state.name}` использованные коллекции: {', '.join(result.routed_collections) or '-'}\n\n"
    return (
        header
        + _GUARD_PREFIX[result.guard_state]
        + result.text
        + _format_sources(result)
        + _format_chunks(result)
    )


def build_demo() -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=respond,
        title="Демо RAG BerserkAI",
        description="Задайте вопрос - ответ построится по найденным фрагментам из различных источников."
                    " (Кстати, под ответом можно раскрыть список использованных чанков)",
        examples=["Как работает атака?", 'Как работает Провокация?', 'Тень Засады'],
        textbox=gr.Textbox(placeholder="Ваш вопрос...", container=False, scale=7),
    )