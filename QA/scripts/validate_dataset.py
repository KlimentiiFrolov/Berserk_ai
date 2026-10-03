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
    "test_id","dataset_version","split","category","subcategory","difficulty",
    "priority","question","expected_answer","expected_facts","answerability",
    "intent","retrieval_target","expected_source_ids","citation_required",
    "temporal","status"
}

def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}:{n}: invalid JSON: {e}")
    return rows

def main():
    errors = []
    rows = load_jsonl(GOLD / "all.jsonl")
    src = json.loads(SOURCES.read_text(encoding="utf-8"))
    source_ids = {x["source_id"] for x in src["sources"]}

    ids = [r.get("test_id") for r in rows]
    duplicates = [x for x, c in Counter(ids).items() if c > 1]
    if duplicates:
        errors.append(f"Duplicate test_id: {duplicates}")

    for r in rows:
        tid = r.get("test_id", "<missing>")
        missing = sorted(REQUIRED - set(r))
        if missing:
            errors.append(f"{tid}: missing fields {missing}")

        for sid in [x.strip() for x in r.get("expected_source_ids", "").split(";") if x.strip()]:
            if sid not in source_ids:
                errors.append(f"{tid}: unknown source id {sid}")

        if r.get("temporal") == "Да" and not r.get("snapshot_date"):
            errors.append(f"{tid}: temporal test without snapshot_date")

    all_by_category = {}
    for r in rows:
        all_by_category.setdefault(r["category"], []).append(r["test_id"])

    for category, filename in CATEGORY_FILES.items():
        split_rows = load_jsonl(GOLD / filename)
        split_ids = [r["test_id"] for r in split_rows]
        expected_ids = all_by_category.get(category, [])
        if split_ids != expected_ids:
            errors.append(f"{filename}: does not match all.jsonl for category {category}")

    if errors:
        print("VALIDATION FAILED")
        for e in errors:
            print(f"- {e}")
        return 1

    print(f"OK: {len(rows)} tests, {len(source_ids)} sources, {len(CATEGORY_FILES)} category files")
    return 0

if __name__ == "__main__":
    sys.exit(main())
