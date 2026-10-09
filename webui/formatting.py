import re
from rag.llm import Message
from rag.schemas import Answer

SNIPPET_LEN = 400
MAX_HISTORY_MESSAGES = 6
_DEBUG_BLOCK_RE = re.compile(r"\n\n<details>.*?</details>\s*$", re.DOTALL)


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, (list, tuple)):
        return " ".join(_to_text(c) for c in content)
    return str(content or "")


def _history_to_messages(history: list) -> list[Message]:
    messages: list[Message] = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            if role in ("user", "assistant"):
                messages.append(Message(role, _to_text(item.get("content"))))
        else:
            user, bot = item
            if user:
                messages.append(Message("user", _to_text(user)))
            if bot:
                messages.append(Message("assistant", _to_text(bot)))

    for m in messages:
        if m.role == "assistant":
            m.content = _DEBUG_BLOCK_RE.sub("", m.content)
    return messages[-MAX_HISTORY_MESSAGES:]


def _format_chunks(result: Answer) -> str:
    if not result.chunks:
        return ""
    lines = []
    for i, rc in enumerate(result.chunks, start=1):
        c = rc.chunk
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > SNIPPET_LEN:
            snippet = snippet[:SNIPPET_LEN] + "..."
        lines.append(
            f"**[{i}]** `{c.collection}` | {c.source_type or 'Нет информации о типе источника'} | "
            f"score={rc.score:.3f} ({rc.stage})\n\n> {snippet}"
        )
    body = "\n\n".join(lines)
    return f"\n\n<details><summary>Найденные фрагменты ({len(result.chunks)})</summary>\n\n{body}\n\n</details>"


def _format_sources(result: Answer) -> str:
    if not result.sources:
        return ""
    items = []
    for s in result.sources:
        label = f"{s.collection}/{s.title} ({s.source_type})"
        items.append(f"- [{label}]({s.url})" if s.url else f"- {label}")
    return "\n\n**Источники:**\n" + "\n".join(items)
