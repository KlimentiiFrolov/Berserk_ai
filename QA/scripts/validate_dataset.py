from pathlib import Path
import json
import sys
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "datasets" / "gold"
SOURCES = ROOT / "sources" / "official_sources.json"

CATEGORY_FILES = {
    "Rules_Basic": "rules_basic.jsonl",
    "Mechanics_Glossary": "mechanics_glossary.jsonl",
    "Cards_Facts": "cards_facts.jsonl",
    "Card_FAQ_Interactions": "card_faq_interactions.jsonl",
    "Tournament_Formats": "tournament_formats.jsonl",
    "Negative_OutOfScope": "negative_out_of_scope.jsonl",
}

REQUIRED = {
    "test_id",
    "dataset_version",
    "split",
    "category",
    "subcategory",
    "difficulty",
    "priority",
    "question",
    "expected_answer",
    "expected_facts",
    "answerability",
    "intent",
    "retrieval_target",
    "expected_source_ids",
    "citation_required",
    "temporal",
    "status",
}


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error

    return rows


def load_source_ids(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {source["source_id"] for source in data["sources"]}


def validate_duplicate_ids(rows):
    ids = [row.get("test_id") for row in rows]

    return [
        f"Duplicate test_id: {test_id}"
        for test_id, count in Counter(ids).items()
        if count > 1
    ]


def validate_required_fields(rows):
    errors = []

    for row in rows:
        test_id = row.get("test_id", "<missing>")
        missing = sorted(REQUIRED - set(row))

        if missing:
            errors.append(f"{test_id}: missing fields {missing}")

    return errors


def validate_source_ids(rows, source_ids):
    errors = []

    for row in rows:
        test_id = row.get("test_id", "<missing>")
        expected_source_ids = row.get("expected_source_ids", "")

        for source_id in expected_source_ids.split(";"):
            source_id = source_id.strip()

            if source_id and source_id not in source_ids:
                errors.append(
                    f"{test_id}: unknown source id {source_id}"
                )

    return errors


def validate_temporal_tests(rows):
    errors = []

    for row in rows:
        test_id = row.get("test_id", "<missing>")

        if row.get("temporal") == "Да" and not row.get("snapshot_date"):
            errors.append(
                f"{test_id}: temporal test without snapshot_date"
            )

    return errors


def group_ids_by_category(rows):
    grouped = {}

    for row in rows:
        category = row["category"]
        grouped.setdefault(category, []).append(row["test_id"])

    return grouped


def validate_category_files(rows):
    errors = []
    expected_by_category = group_ids_by_category(rows)

    for category, filename in CATEGORY_FILES.items():
        split_rows = load_jsonl(GOLD / filename)
        split_ids = [row["test_id"] for row in split_rows]
        expected_ids = expected_by_category.get(category, [])

        if split_ids != expected_ids:
            errors.append(
                f"{filename}: does not match all.jsonl "
                f"for category {category}"
            )

    return errors


def validate_dataset(rows, source_ids):
    errors = []

    errors.extend(validate_duplicate_ids(rows))
    errors.extend(validate_required_fields(rows))
    errors.extend(validate_source_ids(rows, source_ids))
    errors.extend(validate_temporal_tests(rows))
    errors.extend(validate_category_files(rows))

    return errors


def print_validation_errors(errors):
    print("VALIDATION FAILED")

    for error in errors:
        print(f"- {error}")


def main():
    rows = load_jsonl(GOLD / "all.jsonl")
    source_ids = load_source_ids(SOURCES)

    errors = validate_dataset(rows, source_ids)

    if errors:
        print_validation_errors(errors)
        return 1

    print(
        f"OK: {len(rows)} tests, "
        f"{len(source_ids)} sources, "
        f"{len(CATEGORY_FILES)} category files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())