"""Проверка RAG без бота: uv run python -m rag "Как работает атака?" """
import asyncio
import sys

import rag


async def main(question: str) -> None:
    await rag.init()
    try:
        result = await rag.answer(question)
    finally:
        await rag.close()

    print(f"\n[{result.guard_state.name}] collections={result.routed_collections}\n")
    print(result.text)
    if result.sources:
        print("\nИсточники:")
        for source in result.sources:
            print(f"  - {source.collection}/{source.title} ({source.source_type})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Использование: python -m rag "вопрос"')
    asyncio.run(main(" ".join(sys.argv[1:])))
