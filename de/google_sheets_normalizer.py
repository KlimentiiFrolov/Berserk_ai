from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT

RAW_BASE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "google_sheets"
)

PROCESSED_BASE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "google_sheets"
)

CARDS_OUT = (
    PROCESSED_BASE
    / "cards"
)

FAQ_OUT = (
    PROCESSED_BASE
    / "faq"
)

CARD_RELEASE_SHEETS = (
    "Война стихий",
    "Нашествие Тьмы",
    "Легенды Лаара",
    "Семена Раздора",
    "Ложные боги",
    "Путь хранителей",
    "Призрачный легион",
    "Гибель Богов",
    "Легенды Руси",
    "Сердце Роя",
)

FAQ_RELEASE_SHEETS = (
    "7. Гибель богов",
    "6. Призрачный легион",
    "5. Путь хранителей",
    "4. Ложные боги",
    "3. Семена раздора",
    "2. Нашествие тьмы",
    "1. Война стихий",
)


CARD_FIELDS = (
    "number",
    "name",
    "rarity",
    "cost",
    "elite",
    "unique",
    "color",
    "class",
    "type",
    "life",
    "move",
    "hit",
    "text",
    "artist",
    "horde",
    "opp_defense",
    "opp_attack",
    "opp_shot",
    "def_flying",
    "def_poison",
    "def_shot",
    "def_spell",
    "def_magic",
    "def_discharge",
    "def_throw",
    "armor",
    "directed_strike",
    "regeneration",
    "stamina",
    "flavor_text",
)

HEADER_ALIASES = {
    "№": "number",
    "name": "name",
    "rarity": "rarity",
    "cost": "cost",
    "elite": "elite",
    "uniq": "unique",
    "color": "color",
    "class": "class",
    "type": "type",
    "life": "life",
    "move": "move",
    "hit": "hit",
    "text": "text",
    "artist": "artist",
    "орда": "horde",
    "оп в защ": "opp_defense",
    "оп в атак": "opp_attack",
    "оп в стрел": "opp_shot",
    "защ от лет": "def_flying",
    "защ от яда": "def_poison",
    "защ от выстр": "def_shot",
    "защ закл": "def_spell",
    "защ от закл": "def_spell",
    "защ от маг": "def_magic",
    "защ от разр": "def_discharge",
    "защ от мет": "def_throw",
    "броня": "armor",
    "направ удар": "directed_strike",
    "реген": "regeneration",
    "стойкость": "stamina",
    "худ.текст": "flavor_text",
}

NUMERIC_FIELDS = {
    "number",
    "rarity",
    "cost",
    "elite",
    "unique",
    "color",
    "type",
    "life",
    "move",
    "horde",
    "opp_defense",
    "opp_attack",
    "opp_shot",
    "def_flying",
    "def_poison",
    "def_shot",
    "def_spell",
    "def_magic",
    "def_discharge",
    "def_throw",
    "armor",
    "directed_strike",
    "regeneration",
    "stamina",
}


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as fh:
        for line in fh:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as fh:
        for record in records:
            fh.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def clean_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(value)

    # Excel escaped control characters seen in source.
    text = re.sub(
        r"_x000[0-9A-Fa-f]_",
        " ",
        text,
    )

    text = text.replace(
        "\u00a0",
        " ",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    text = text.strip()

    return text or None


def normalize_number(
    value: Any,
) -> int | float | str | None:
    text = clean_text(value)

    if text is None:
        return None

    if re.fullmatch(
        r"-?\d+",
        text,
    ):
        return int(text)

    if re.fullmatch(
        r"-?\d+\.0+",
        text,
    ):
        return int(
            float(text)
        )

    if re.fullmatch(
        r"-?\d+\.\d+",
        text,
    ):
        return float(text)

    return text


def normalize_code_key(
    value: Any,
) -> str | None:
    normalized = normalize_number(
        value
    )

    if normalized is None:
        return None

    return str(normalized)


def normalize_hit(
    value: Any,
) -> tuple[str | None, bool]:
    """
    Some old workbook cells like 1-2-3 were interpreted by Excel as
    dates and cached as 2003-02-01. For hit only, reconstruct D-M-YY.
    """
    text = clean_text(value)

    if text is None:
        return None, False

    iso_match = re.fullmatch(
        r"(\d{4})-(\d{2})-(\d{2})(?:T.*)?",
        text,
    )

    if iso_match:
        year = int(
            iso_match.group(1)
        )
        month = int(
            iso_match.group(2)
        )
        day = int(
            iso_match.group(3)
        )

        if (
            2000 <= year <= 2099
            and 1 <= month <= 12
            and 1 <= day <= 31
        ):
            return (
                f"{day}-{month}-{year % 100}",
                True,
            )

    return text, False


def normalize_boolish(
    value: Any,
) -> bool | None:
    text = clean_text(value)

    if text is None:
        return None

    lowered = text.casefold()

    if lowered in {
        "true",
        "1",
        "1.0",
        "yes",
        "да",
    }:
        return True

    if lowered in {
        "false",
        "0",
        "0.0",
        "no",
        "нет",
    }:
        return False

    return None


def slugify(value: str) -> str:
    value = value.casefold()
    value = re.sub(
        r"[^0-9a-zа-яё]+",
        "-",
        value,
    )
    return value.strip("-")


def stable_id(
    prefix: str,
    *parts: Any,
) -> str:
    raw = "|".join(
        str(part or "")
        for part in parts
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"{prefix}_{digest}"
    )


def workbook_manifest(
    source_key: str,
) -> dict[str, Any]:
    path = (
        RAW_BASE
        / source_key
        / "workbook_sheets"
        / "workbook_manifest.json"
    )

    return load_json(path)


def sheet_rows_by_name(
    source_key: str,
) -> dict[str, list[dict[str, Any]]]:
    manifest = workbook_manifest(
        source_key
    )

    result = {}

    for sheet in manifest["sheets"]:
        result[
            sheet["sheet_name"]
        ] = load_jsonl(
            Path(
                sheet[
                    "rows_jsonl"
                ]
            )
        )

    return result


def parse_glossary(
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, str]],
]:
    records = []

    maps: dict[
        str,
        dict[str, str],
    ] = {
        "color": {},
        "rarity": {},
        "type": {},
    }

    middle_section = None
    right_section = None

    for row in rows[1:]:
        values = list(
            row.get("values") or []
        )

        values += [""] * (
            12 - len(values)
        )

        symbol_name = clean_text(
            values[1]
        )
        tag = clean_text(
            values[2]
        )
        proberserk = clean_text(
            values[3]
        )

        if symbol_name or tag:
            records.append(
                {
                    "glossary_id":
                        stable_id(
                            "glossary",
                            "symbol",
                            row[
                                "row_number"
                            ],
                            symbol_name,
                            tag,
                        ),
                    "kind": "symbol",
                    "name":
                        symbol_name,
                    "tag":
                        tag,
                    "external_value":
                        proberserk,
                    "source": {
                        "sheet":
                            "Обозначения",
                        "row":
                            row[
                                "row_number"
                            ],
                    },
                }
            )

        middle_name = clean_text(
            values[7]
        )
        middle_code = (
            normalize_code_key(
                values[8]
            )
        )

        if (
            middle_name
            and middle_code is None
            and middle_name.endswith(":")
        ):
            middle_section = (
                middle_name
                .rstrip(":")
                .strip()
            )

        elif (
            middle_name
            and middle_code is not None
        ):
            group = (
                middle_section
                or "mapping"
            )

            records.append(
                {
                    "glossary_id":
                        stable_id(
                            "glossary",
                            group,
                            middle_code,
                            middle_name,
                        ),
                    "kind": "code_mapping",
                    "group":
                        group,
                    "name":
                        middle_name,
                    "code":
                        middle_code,
                    "source": {
                        "sheet":
                            "Обозначения",
                        "row":
                            row[
                                "row_number"
                            ],
                    },
                }
            )

            lowered = (
                group.casefold()
            )

            if "стих" in lowered:
                maps[
                    "color"
                ][middle_code] = (
                    middle_name
                )

            elif "редк" in lowered:
                maps[
                    "rarity"
                ][middle_code] = (
                    middle_name
                )

        right_name = clean_text(
            values[10]
        )
        right_code = (
            normalize_code_key(
                values[11]
            )
        )

        if (
            right_name
            and right_code is None
            and right_name.endswith(":")
        ):
            right_section = (
                right_name
                .rstrip(":")
                .strip()
            )

        elif (
            right_name
            and right_code is not None
        ):
            group = (
                right_section
                or "mapping"
            )

            records.append(
                {
                    "glossary_id":
                        stable_id(
                            "glossary",
                            group,
                            right_code,
                            right_name,
                        ),
                    "kind": "code_mapping",
                    "group":
                        group,
                    "name":
                        right_name,
                    "code":
                        right_code,
                    "source": {
                        "sheet":
                            "Обозначения",
                        "row":
                            row[
                                "row_number"
                            ],
                    },
                }
            )

            if "тип" in (
                group.casefold()
            ):
                maps[
                    "type"
                ][right_code] = (
                    right_name
                )

    return records, maps


def card_header_map(
    header_values: list[Any],
) -> dict[int, str]:
    mapping = {}

    for index, raw in enumerate(
        header_values
    ):
        header = (
            clean_text(raw)
            or ""
        ).casefold()

        field = HEADER_ALIASES.get(
            header
        )

        if field:
            mapping[index] = field

    return mapping


def normalize_cards(
    sheets: dict[
        str,
        list[dict[str, Any]],
    ],
    glossary_maps: dict[
        str,
        dict[str, str],
    ],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    cards = []
    rejected = []

    hit_dates_fixed = 0
    header_alias_sheets = []

    for release in CARD_RELEASE_SHEETS:
        rows = sheets.get(release)

        if not rows:
            raise RuntimeError(
                f"Нет листа карт: {release}"
            )

        header_values = list(
            rows[0].get(
                "values"
            )
            or []
        )

        mapping = card_header_map(
            header_values
        )

        if (
            "защ от закл"
            in [
                (
                    clean_text(v)
                    or ""
                ).casefold()
                for v in header_values
            ]
        ):
            header_alias_sheets.append(
                release
            )

        missing = (
            set(CARD_FIELDS)
            - set(mapping.values())
        )

        if missing:
            raise RuntimeError(
                f"{release}: не хватает полей "
                f"{sorted(missing)}"
            )

        for row in rows[1:]:
            values = list(
                row.get("values") or []
            )

            data = {
                field: (
                    values[index]
                    if index < len(values)
                    else None
                )
                for index, field
                in mapping.items()
            }

            name = clean_text(
                data.get("name")
            )

            if not name:
                if any(
                    clean_text(value)
                    for value in values
                ):
                    rejected.append(
                        {
                            "reason":
                                "missing_card_name",
                            "release":
                                release,
                            "source_row":
                                row[
                                    "row_number"
                                ],
                            "values":
                                values,
                        }
                    )
                continue

            normalized = {}

            for field in CARD_FIELDS:
                raw_value = data.get(
                    field
                )

                if field == "hit":
                    (
                        normalized[
                            field
                        ],
                        fixed,
                    ) = normalize_hit(
                        raw_value
                    )

                    if fixed:
                        hit_dates_fixed += 1

                elif field in NUMERIC_FIELDS:
                    normalized[
                        field
                    ] = normalize_number(
                        raw_value
                    )

                else:
                    normalized[
                        field
                    ] = clean_text(
                        raw_value
                    )

            rarity_key = (
                normalize_code_key(
                    normalized[
                        "rarity"
                    ]
                )
            )
            color_key = (
                normalize_code_key(
                    normalized[
                        "color"
                    ]
                )
            )
            type_key = (
                normalize_code_key(
                    normalized[
                        "type"
                    ]
                )
            )

            normalized[
                "rarity_name"
            ] = glossary_maps[
                "rarity"
            ].get(
                rarity_key or ""
            )

            normalized[
                "color_name"
            ] = glossary_maps[
                "color"
            ].get(
                color_key or ""
            )

            normalized[
                "type_name"
            ] = glossary_maps[
                "type"
            ].get(
                type_key or ""
            )

            number = normalized[
                "number"
            ]

            card_id = stable_id(
                "card",
                release,
                number,
                name,
            )

            retrieval_parts = [
                f"Карта: {name}",
                f"Выпуск: {release}",
            ]

            if normalized[
                "cost"
            ] is not None:
                retrieval_parts.append(
                    "Стоимость: "
                    f"{normalized['cost']}"
                )

            if normalized[
                "life"
            ] is not None:
                retrieval_parts.append(
                    "Жизни: "
                    f"{normalized['life']}"
                )

            if normalized[
                "move"
            ] is not None:
                retrieval_parts.append(
                    "Движение: "
                    f"{normalized['move']}"
                )

            if normalized[
                "hit"
            ]:
                retrieval_parts.append(
                    "Удар: "
                    f"{normalized['hit']}"
                )

            if normalized[
                "class"
            ]:
                retrieval_parts.append(
                    "Класс: "
                    f"{normalized['class']}"
                )

            if normalized[
                "type_name"
            ]:
                retrieval_parts.append(
                    "Тип: "
                    f"{normalized['type_name']}"
                )

            if normalized[
                "text"
            ]:
                retrieval_parts.append(
                    "Текст карты: "
                    f"{normalized['text']}"
                )

            if normalized[
                "flavor_text"
            ]:
                retrieval_parts.append(
                    "Художественный текст: "
                    f"{normalized['flavor_text']}"
                )

            cards.append(
                {
                    "schema_version": 1,
                    "card_id":
                        card_id,
                    "source_type":
                        "card_database",
                    "authority":
                        "structured_reference",
                    "authority_rank":
                        90,
                    "release":
                        release,
                    **normalized,
                    "retrieval_text":
                        "\n".join(
                            retrieval_parts
                        ),
                    "source": {
                        "workbook":
                            "cards",
                        "sheet":
                            release,
                        "row":
                            row[
                                "row_number"
                            ],
                    },
                }
            )

    stats = {
        "cards":
            len(cards),
        "rejected_rows":
            len(rejected),
        "hit_dates_reconstructed":
            hit_dates_fixed,
        "header_alias_sheets":
            header_alias_sheets,
    }

    return cards, rejected, stats


def approval_status(
    value: Any,
) -> tuple[
    str,
    int,
]:
    parsed = normalize_boolish(
        value
    )

    if parsed is True:
        return (
            "approved",
            100,
        )

    if parsed is False:
        return (
            "not_approved_in_faq",
            85,
        )

    return (
        "unknown_legacy",
        75,
    )


def release_name_from_faq_sheet(
    sheet_name: str,
) -> str:
    return re.sub(
        r"^\d+\.\s*",
        "",
        sheet_name,
    ).strip()


def normalize_faq(
    sheets: dict[
        str,
        list[dict[str, Any]],
    ],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    records = []
    unanswered = []

    approved_count = 0
    unapproved_count = 0
    unknown_count = 0
    qna_count = 0
    ruling_count = 0

    for sheet_name in FAQ_RELEASE_SHEETS:
        rows = sheets.get(
            sheet_name
        )

        if not rows:
            raise RuntimeError(
                f"Нет FAQ-листа: "
                f"{sheet_name}"
            )

        release = (
            release_name_from_faq_sheet(
                sheet_name
            )
        )

        current_number = None
        current_title = None

        for row in rows[1:]:
            values = list(
                row.get("values")
                or []
            )

            values += [""] * (
                5 - len(values)
            )

            number = normalize_number(
                values[0]
            )
            title = clean_text(
                values[1]
            )
            question = clean_text(
                values[2]
            )
            answer = clean_text(
                values[3]
            )

            if (
                title
                and title.startswith(
                    "Это FAQ "
                )
            ):
                continue

            if number is not None:
                current_number = number

            if title:
                current_title = title

            status, rank = (
                approval_status(
                    values[4]
                )
            )

            if not question and not answer:
                continue

            base_record = {
                "schema_version": 1,
                "source_type":
                    "faq",
                "release":
                    release,
                "card_number":
                    current_number,
                "card_name":
                    current_title,
                "question":
                    question,
                "answer":
                    answer,
                "approval_status":
                    status,
                "approved":
                    (
                        True
                        if status
                        == "approved"
                        else (
                            False
                            if status
                            == "not_approved_in_faq"
                            else None
                        )
                    ),
                "authority":
                    (
                        "faq_approved"
                        if status
                        == "approved"
                        else (
                            "faq_not_approved"
                            if status
                            == "not_approved_in_faq"
                            else
                            "faq_legacy_unknown"
                        )
                    ),
                "authority_rank":
                    rank,
                "source": {
                    "workbook":
                        "faq",
                    "sheet":
                        sheet_name,
                    "row":
                        row[
                            "row_number"
                        ],
                },
            }

            if status == "approved":
                approved_count += 1
            elif (
                status
                == "not_approved_in_faq"
            ):
                unapproved_count += 1
            else:
                unknown_count += 1

            if question and not answer:
                unanswered.append(
                    {
                        **base_record,
                        "faq_id":
                            stable_id(
                                "faq_unanswered",
                                sheet_name,
                                row[
                                    "row_number"
                                ],
                                question,
                            ),
                        "kind":
                            "unanswered",
                    }
                )
                continue

            if question:
                kind = "qna"
                qna_count += 1
            else:
                kind = "ruling"
                ruling_count += 1

            faq_id = stable_id(
                "faq",
                sheet_name,
                row["row_number"],
                question,
                answer,
            )

            retrieval_parts = [
                f"FAQ: {release}",
            ]

            if current_title:
                retrieval_parts.append(
                    "Карта: "
                    f"{current_title}"
                )

            if question:
                retrieval_parts.append(
                    "Вопрос: "
                    f"{question}"
                )

            if answer:
                retrieval_parts.append(
                    "Ответ: "
                    f"{answer}"
                )

            records.append(
                {
                    **base_record,
                    "faq_id":
                        faq_id,
                    "kind":
                        kind,
                    "retrieval_text":
                        "\n".join(
                            retrieval_parts
                        ),
                }
            )

    stats = {
        "faq_records":
            len(records),
        "qna":
            qna_count,
        "rulings":
            ruling_count,
        "unanswered":
            len(unanswered),
        "approved":
            approved_count,
        "not_approved_in_faq":
            unapproved_count,
        "unknown_legacy":
            unknown_count,
    }

    return records, unanswered, stats


def normalize_checklist(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []

    current_heading = None
    current_event = None

    for row in rows[1:]:
        values = list(
            row.get("values")
            or []
        )

        values += [""] * (
            2 - len(values)
        )

        left = clean_text(
            values[0]
        )
        right = clean_text(
            values[1]
        )

        if not left and not right:
            continue

        if (
            left
            and not right
        ):
            kind = (
                "intro"
                if len(left) > 180
                else "heading"
            )

            if kind == "heading":
                current_heading = left
                current_event = left

            record_id = stable_id(
                "checklist",
                row["row_number"],
                left,
            )

            records.append(
                {
                    "schema_version": 1,
                    "checklist_id":
                        record_id,
                    "source_type":
                        "game_checklist",
                    "authority":
                        "official_or_editorial_reference",
                    "authority_rank":
                        90,
                    "kind":
                        kind,
                    "heading":
                        (
                            left
                            if kind
                            == "heading"
                            else None
                        ),
                    "event":
                        None,
                    "action":
                        None,
                    "text":
                        left,
                    "retrieval_text":
                        left,
                    "source": {
                        "workbook":
                            "faq",
                        "sheet":
                            "Чек-лист на игру",
                        "row":
                            row[
                                "row_number"
                            ],
                    },
                }
            )

            continue

        if left:
            current_event = left

        event = (
            left
            or current_event
        )

        retrieval = (
            f"Чек-лист игры"
            + (
                f"\nРаздел: {current_heading}"
                if current_heading
                else ""
            )
            + (
                f"\nСобытие: {event}"
                if event
                else ""
            )
            + (
                f"\nДействие: {right}"
                if right
                else ""
            )
        )

        records.append(
            {
                "schema_version": 1,
                "checklist_id":
                    stable_id(
                        "checklist",
                        row[
                            "row_number"
                        ],
                        event,
                        right,
                    ),
                "source_type":
                    "game_checklist",
                "authority":
                    "official_or_editorial_reference",
                "authority_rank":
                    90,
                "kind":
                    "action",
                "heading":
                    current_heading,
                "event":
                    event,
                "action":
                    right,
                "text":
                    right,
                "retrieval_text":
                    retrieval,
                "source": {
                    "workbook":
                        "faq",
                    "sheet":
                        "Чек-лист на игру",
                    "row":
                        row[
                            "row_number"
                        ],
                },
            }
        )

    return records


def normalize_tournament_notes(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    current_heading = None

    for row in rows:
        values = list(
            row.get("values")
            or []
        )

        values += [""] * (
            2 - len(values)
        )

        left = clean_text(
            values[0]
        )
        right = clean_text(
            values[1]
        )

        if not left and not right:
            continue

        if (
            left
            and not right
            and len(left) <= 160
        ):
            current_heading = left

            records.append(
                {
                    "schema_version": 1,
                    "note_id":
                        stable_id(
                            "tournament_heading",
                            row[
                                "row_number"
                            ],
                            left,
                        ),
                    "source_type":
                        "tournament_note",
                    "authority":
                        "official_or_editorial_reference",
                    "authority_rank":
                        95,
                    "kind":
                        "heading",
                    "heading":
                        left,
                    "body":
                        None,
                    "detail":
                        None,
                    "retrieval_text":
                        left,
                    "source": {
                        "workbook":
                            "faq",
                        "sheet":
                            "Важные турнирные моменты",
                        "row":
                            row[
                                "row_number"
                            ],
                    },
                }
            )

            continue

        retrieval_parts = [
            "Важные турнирные моменты",
        ]

        if current_heading:
            retrieval_parts.append(
                current_heading
            )

        if left:
            retrieval_parts.append(
                left
            )

        if right:
            retrieval_parts.append(
                right
            )

        records.append(
            {
                "schema_version": 1,
                "note_id":
                    stable_id(
                        "tournament_note",
                        row[
                            "row_number"
                        ],
                        current_heading,
                        left,
                        right,
                    ),
                "source_type":
                    "tournament_note",
                "authority":
                    "official_or_editorial_reference",
                "authority_rank":
                    95,
                "kind":
                    "note",
                "heading":
                    current_heading,
                "body":
                    left,
                "detail":
                    right,
                "retrieval_text":
                    "\n".join(
                        retrieval_parts
                    ),
                "source": {
                    "workbook":
                        "faq",
                    "sheet":
                        "Важные турнирные моменты",
                    "row":
                        row[
                            "row_number"
                        ],
                },
            }
        )

    return records


def preview(
    records: list[dict[str, Any]],
    limit: int = 25,
) -> list[dict[str, Any]]:
    return records[:limit]


def run() -> dict[str, Any]:
    CARDS_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FAQ_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    card_sheets = (
        sheet_rows_by_name(
            "cards"
        )
    )

    faq_sheets = (
        sheet_rows_by_name(
            "faq"
        )
    )

    glossary, glossary_maps = (
        parse_glossary(
            card_sheets[
                "Обозначения"
            ]
        )
    )

    (
        cards,
        rejected_cards,
        card_stats,
    ) = normalize_cards(
        card_sheets,
        glossary_maps,
    )

    (
        faq,
        unanswered,
        faq_stats,
    ) = normalize_faq(
        faq_sheets
    )

    checklist = normalize_checklist(
        faq_sheets[
            "Чек-лист на игру"
        ]
    )

    tournament_notes = (
        normalize_tournament_notes(
            faq_sheets[
                "Важные турнирные моменты"
            ]
        )
    )

    write_jsonl(
        CARDS_OUT
        / "cards.jsonl",
        cards,
    )

    write_jsonl(
        CARDS_OUT
        / "glossary.jsonl",
        glossary,
    )

    write_jsonl(
        CARDS_OUT
        / "cards_rejected.jsonl",
        rejected_cards,
    )

    write_jsonl(
        FAQ_OUT
        / "faq.jsonl",
        faq,
    )

    write_jsonl(
        FAQ_OUT
        / "faq_unanswered.jsonl",
        unanswered,
    )

    write_jsonl(
        FAQ_OUT
        / "checklist.jsonl",
        checklist,
    )

    write_jsonl(
        FAQ_OUT
        / "tournament_notes.jsonl",
        tournament_notes,
    )

    (
        CARDS_OUT
        / "cards_preview.json"
    ).write_text(
        json.dumps(
            preview(cards),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        FAQ_OUT
        / "faq_preview.json"
    ).write_text(
        json.dumps(
            preview(faq),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "cards": {
            **card_stats,
            "glossary_records":
                len(glossary),
            "output":
                str(CARDS_OUT),
        },
        "faq": {
            **faq_stats,
            "checklist_records":
                len(checklist),
            "tournament_note_records":
                len(
                    tournament_notes
                ),
            "output":
                str(FAQ_OUT),
        },
        "authority_policy": {
            "faq_approved": 100,
            "tournament_note": 95,
            "card_database": 90,
            "game_checklist": 90,
            "faq_not_approved": 85,
            "faq_legacy_unknown": 75,
            "telegram_community":
                "lower; assigned later",
        },
    }

    summary_path = (
        PROCESSED_BASE
        / "google_sheets_normalization_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("GOOGLE SHEETS NORMALIZATION DONE")
    print("=" * 72)
    print(
        f"Cards:              "
        f"{len(cards)}"
    )
    print(
        f"Rejected card rows: "
        f"{len(rejected_cards)}"
    )
    print(
        f"Hit dates repaired: "
        f"{card_stats['hit_dates_reconstructed']}"
    )
    print(
        f"Glossary:           "
        f"{len(glossary)}"
    )
    print(
        f"FAQ knowledge:      "
        f"{len(faq)}"
    )
    print(
        f"FAQ unanswered:     "
        f"{len(unanswered)}"
    )
    print(
        f"Checklist:          "
        f"{len(checklist)}"
    )
    print(
        f"Tournament notes:   "
        f"{len(tournament_notes)}"
    )
    print(
        f"Summary: {summary_path}"
    )

    return summary
