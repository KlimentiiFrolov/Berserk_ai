from rag.schemas import Answer, AnswerGuardState, RetrievedChunk, Source


def sources_from_chunks(chunks: list[RetrievedChunk]) -> list[Source]:
    sources: list[Source] = []
    seen: set[tuple[str, str, str | None]] = set()

    for retrieved in chunks:
        chunk = retrieved.chunk
        payload = chunk.metadata
        citation = payload.get("citation") or {}
        provenance = payload.get("provenance") or {}

        source_id = payload.get("source_id")
        title = (
            payload.get("title")
            or payload.get("section_path")
            or payload.get("card_name")
            or payload.get("question")
            or chunk.id
        )
        url = (
            payload.get("source_url")
            or citation.get("url")
            or provenance.get("url")
        )
        source_type = "community" if (
            chunk.collection == "community"
            or payload.get("source_type") in {"community", "community_qa"}
        ) else "official"
        key = (chunk.collection, str(source_id or title), url)
        if key in seen:
            continue
        seen.add(key)
        sources.append(Source(
            collection=chunk.collection,
            source_type=source_type,
            title=str(title),
            url=str(url) if url else None,
            source_id=str(source_id) if source_id else None,
            chunk_id=chunk.id,
        ))

    return sources