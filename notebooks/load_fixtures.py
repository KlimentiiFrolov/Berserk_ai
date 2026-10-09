from __future__ import annotations
from pathlib import Path
import yaml

from rag.schemas import Chunk


class FixtureValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Ошибки в fixture-данных:\n" + "\n".join(f"  - {e}" for e in errors))


def _require_fields(record: dict, fields: list[str], record_label: str, errors: list[str]) -> None:
    for f in fields:
        if f not in record or record[f] in (None, ""):
            errors.append(f"{record_label}: отсутствует обязательное поле '{f}'")


def _check_unique_ids(records: list[dict], id_field: str, collection: str, errors: list[str]) -> None:
    seen: set[str] = set()
    for r in records:
        rid = r.get(id_field)
        if rid is None:
            continue
        if rid in seen:
            errors.append(f"{collection}: дублирующийся {id_field}='{rid}'")
        seen.add(rid)


def load_rules(path: str | Path) -> list[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    records = data["rules"]

    for r in records:
        if "chunk_id" in r:
            r["id"] = r.pop("chunk_id")

    errors: list[str] = []
    _check_unique_ids(records, "id", "rules", errors)
    for r in records:
        _require_fields(r, ["id", "section_path", "text", "source_type", "doc_version"],
                         f"rules[{r.get('id', '?')}]", errors)
    if errors:
        raise FixtureValidationError(errors)

    return [
        Chunk(
            id=r["id"],
            text=r["text"],
            source_type=r["source_type"],
            collection="rules",
            metadata=r,
        )
        for r in records
    ]


def load_cards(path: str | Path) -> list[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    records = data["cards"]

    for r in records:
        if "card_id" in r:
            r["id"] = r.pop("card_id")

    errors: list[str] = []
    _check_unique_ids(records, "id", "cards", errors)
    for r in records:
        _require_fields(r, ["id", "name", "text", "source_type"],
                         f"cards[{r.get('id', '?')}]", errors)
    if errors:
        raise FixtureValidationError(errors)

    chunks = []
    for r in records:
        r["text"] = f"{r['name']}. {r['text']}"
        chunks.append(Chunk(
            id=r["id"],
            text=r["text"],
            source_type=r["source_type"],
            collection="cards",
            metadata=r,
        ))
    return chunks


def load_faq(path: str | Path) -> list[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    records = data["faq"]

    for r in records:
        if "faq_id" in r:
            r["id"] = r.pop("faq_id")

    errors: list[str] = []
    _check_unique_ids(records, "id", "faq", errors)
    for r in records:
        _require_fields(r, ["id", "question", "answer", "source_type"],
                         f"faq[{r.get('id', '?')}]", errors)
    if errors:
        raise FixtureValidationError(errors)

    chunks = []
    for r in records:
        r["text"] = r["question"] + " " + r.get("answer", "")
        chunks.append(Chunk(
            id=r["id"],
            text=r["text"],
            source_type=r["source_type"],
            collection="faq",
            metadata=r,
        ))
    return chunks


def load_community(path: str | Path) -> list[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    records = data["community"]

    for r in records:
        if "chunk_id" in r:
            r["id"] = r.pop("chunk_id")

    errors: list[str] = []
    _check_unique_ids(records, "id", "community", errors)
    for r in records:
        _require_fields(r, ["id", "text", "source_url", "source_type", "date"],
                         f"community[{r.get('id', '?')}]", errors)
    if errors:
        raise FixtureValidationError(errors)

    return [
        Chunk(
            id=r["id"],
            text=r["text"],
            source_type=r["source_type"],
            collection="community",
            metadata=r,
        )
        for r in records
    ]


_LOADERS = {
    "rules": load_rules,
    "cards": load_cards,
    "faq": load_faq,
    "community": load_community,
}


def load_all_fixtures(fixtures_dir: str | Path) -> dict[str, list[Chunk]]:
    fixtures_dir = Path(fixtures_dir)
    result = {}
    all_errors: list[str] = []
    for collection, loader in _LOADERS.items():
        path = fixtures_dir / f"{collection}.yaml"
        try:
            result[collection] = loader(path)
        except FixtureValidationError as e:
            all_errors.extend(e.errors)
    if all_errors:
        raise FixtureValidationError(all_errors)
    return result