from pathlib import Path
import json
import sys
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "datasets" / "robustness"

FILES = [
    ROBUSTNESS / "paraphrase_pairs.jsonl",
    ROBUSTNESS / "false_premise.jsonl",
]

REQUIRED_COMMON = {
    "test_id",
    "dataset_version",
    "split",
    "variant_type",
    "question",
    "expected_facts",
    "expected_source_ids",
    "evaluation_focus",
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


def validate_rows(rows, filename):
    errors = []

    ids = [row.get("test_id") for row in rows]
    for test_id, count in Counter(ids).items():
        if count > 1:
            errors.append(f"{filename}: duplicate test_id {test_id}")

    for row in rows:
        test_id = row.get("test_id", "<missing>")
        missing = sorted(REQUIRED_COMMON - set(row))

        if missing:
            errors.append(f"{test_id}: missing fields {missing}")

        if not isinstance(row.get("expected_facts"), list) or not row.get("expected_facts"):
            errors.append(f"{test_id}: expected_facts must be a non-empty list")

        if not isinstance(row.get("expected_source_ids"), list) or not row.get("expected_source_ids"):
            errors.append(
                f"{test_id}: expected_source_ids must be a non-empty list"
            )

        if row.get("split") != "robustness":
            errors.append(f"{test_id}: split must be 'robustness'")

        if row.get("variant_type") == "paraphrase" and not row.get("source_test_id"):
            errors.append(f"{test_id}: paraphrase case requires source_test_id")

        if row.get("variant_type") == "false_premise" and not row.get("correction_expected"):
            errors.append(
                f"{test_id}: false_premise case requires correction_expected"
            )

    return errors


def main():
    all_errors = []
    total = 0

    for path in FILES:
        if not path.exists():
            all_errors.append(f"Missing file: {path}")
            continue

        rows = load_jsonl(path)
        total += len(rows)
        all_errors.extend(validate_rows(rows, path.name))

    if all_errors:
        print("ROBUSTNESS VALIDATION FAILED")
        for error in all_errors:
            print(f"- {error}")
        return 1

    print(f"OK: {total} robustness tests across {len(FILES)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
