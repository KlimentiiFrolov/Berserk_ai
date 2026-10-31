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

    print(f"\nВопрос: {question}")
    print(f"[{result.guard_state.name}] collections={result.routed_collections}\n")
    print(result.text)

    if result.sources:
        print("\nИсточники:")
        for source in result.sources:
            print(f"  - {source.collection}: {source.title}")
            print(f"    type={source.source_type} source_id={source.source_id or '-'}")
            print(f"    chunk_id={source.chunk_id or '-'}")
            if source.url:
                print(f"    url={source.url}")

    if result.chunks:
        print("\nНайденные чанки:")
        for rank, retrieved in enumerate(result.chunks, start=1):
            chunk = retrieved.chunk
            print(
                f"  {rank}. {chunk.collection}/{chunk.id} "
                f"score={retrieved.score:.4f} stage={retrieved.stage}"
            )
            print(f"     title={chunk.metadata.get('title') or '-'}")
            print(f"     text={chunk.text or '[пустой текст]'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Использование: python -m rag "вопрос"')
    asyncio.run(main(" ".join(sys.argv[1:])))
