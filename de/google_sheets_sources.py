from __future__ import annotations

import csv
import hashlib
import json
import string
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import PROJECT_ROOT

RAW_BASE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "google_sheets"
)


@dataclass(frozen=True)
class SheetSource:
    key: str
    title: str
    spreadsheet_id: str
    gid: str
    authority: str
    source_type: str

    @property
    def edit_url(self) -> str:
        return (
            "https://docs.google.com/spreadsheets/d/"
            f"{self.spreadsheet_id}/edit?gid={self.gid}"
        )

    @property
    def xlsx_url(self) -> str:
        return (
            "https://docs.google.com/spreadsheets/d/"
            f"{self.spreadsheet_id}/export?format=xlsx"
        )

    @property
    def csv_url(self) -> str:
        return (
            "https://docs.google.com/spreadsheets/d/"
            f"{self.spreadsheet_id}/export"
            f"?format=csv&gid={self.gid}"
        )

    @property
    def gviz_csv_url(self) -> str:
        return (
            "https://docs.google.com/spreadsheets/d/"
            f"{self.spreadsheet_id}/gviz/tq"
            f"?tqx=out:csv&gid={self.gid}"
        )


SOURCES = (
    SheetSource(
        key="cards",
        title="База карт",
        spreadsheet_id=(
            "1PLETyADZnX1w_0TkznucDys0qoN82IWSc5HNdxdU1Iw"
        ),
        gid="186452828",
        authority="structured_reference",
        source_type="card_database",
    ),
    SheetSource(
        key="faq",
        title="FAQ",
        spreadsheet_id=(
            "1hSmaM_1SzxaWbWaIkPN4sgobjLWoACD97W8H_nhQ4WY"
        ),
        gid="522308198",
        authority="official_or_editorial_faq",
        source_type="faq",
    ),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_bytes(
    url: str,
    *,
    timeout: int = 120,
) -> tuple[bytes, dict[str, str]]:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 BerserkRAG/1.0 "
                "(Google Sheets RAW collector)"
            ),
            "Accept": "*/*",
        },
    )

    try:
        with urlopen(
            request,
            timeout=timeout,
        ) as response:
            data = response.read()

            headers = {
                str(key).lower(): str(value)
                for key, value
                in response.headers.items()
            }

            return data, headers

    except HTTPError as exc:
        body = exc.read()[:500]
        raise RuntimeError(
            f"HTTP {exc.code} при скачивании {url}. "
            f"Response preview: {body!r}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            f"Не удалось скачать {url}: {exc}"
        ) from exc


def looks_like_html(
    data: bytes,
    headers: dict[str, str],
) -> bool:
    content_type = headers.get(
        "content-type",
        "",
    ).lower()

    prefix = data[:200].lstrip().lower()

    return (
        "text/html" in content_type
        or prefix.startswith(b"<!doctype html")
        or prefix.startswith(b"<html")
    )


def fetch_csv_with_fallback(
    source: SheetSource,
) -> tuple[
    bytes,
    dict[str, str],
    str,
]:
    attempts = (
        source.csv_url,
        source.gviz_csv_url,
    )

    errors: list[str] = []

    for url in attempts:
        try:
            data, headers = fetch_bytes(url)

            if looks_like_html(
                data,
                headers,
            ):
                errors.append(
                    f"{url}: вместо CSV получен HTML"
                )
                continue

            return data, headers, url

        except Exception as exc:
            errors.append(
                f"{url}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "Не удалось получить CSV вкладки.\n"
        + "\n".join(errors)
        + "\nПроверь публичный доступ к Google Sheet."
    )


def decode_csv(
    data: bytes,
) -> str:
    # Google normally returns UTF-8. BOM is handled too.
    return data.decode(
        "utf-8-sig",
        errors="strict",
    )


def column_name(
    index_zero_based: int,
) -> str:
    value = index_zero_based + 1
    result = ""

    while value:
        value, rem = divmod(
            value - 1,
            26,
        )
        result = (
            chr(ord("A") + rem)
            + result
        )

    return result


def parse_csv_rows(
    csv_text: str,
) -> list[list[str]]:
    reader = csv.reader(
        csv_text.splitlines()
    )

    return [
        list(row)
        for row in reader
    ]


def normalize_width(
    rows: list[list[str]],
) -> tuple[list[list[str]], int]:
    width = max(
        (len(row) for row in rows),
        default=0,
    )

    normalized = [
        row + [""] * (
            width - len(row)
        )
        for row in rows
    ]

    return normalized, width


def row_is_blank(
    row: list[str],
) -> bool:
    return all(
        not str(value).strip()
        for value in row
    )


def write_rows_jsonl(
    path: Path,
    rows: list[list[str]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as fh:
        for row_number, row in enumerate(
            rows,
            start=1,
        ):
            by_column = {
                column_name(index): value
                for index, value
                in enumerate(row)
            }

            record = {
                "row_number": row_number,
                "blank": row_is_blank(row),
                "values": row,
                "by_column": by_column,
            }

            fh.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def make_preview(
    rows: list[list[str]],
    *,
    max_rows: int = 20,
) -> list[dict[str, Any]]:
    preview: list[
        dict[str, Any]
    ] = []

    for row_number, row in enumerate(
        rows[:max_rows],
        start=1,
    ):
        preview.append(
            {
                "row_number": row_number,
                "blank": row_is_blank(row),
                "values": row,
            }
        )

    return preview


def collect_source(
    source: SheetSource,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source_dir = (
        RAW_BASE
        / source.key
    )

    source_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    xlsx_path = (
        source_dir
        / "spreadsheet.xlsx"
    )
    csv_path = (
        source_dir
        / f"sheet_{source.gid}.csv"
    )
    rows_path = (
        source_dir
        / f"sheet_{source.gid}_rows.jsonl"
    )
    metadata_path = (
        source_dir
        / "source.json"
    )
    preview_path = (
        source_dir
        / f"sheet_{source.gid}_preview.json"
    )

    if (
        not force
        and metadata_path.exists()
        and xlsx_path.exists()
        and csv_path.exists()
        and rows_path.exists()
    ):
        print(
            f"[SKIP] {source.title}: "
            "RAW уже существует "
            "(используй --force для обновления)"
        )

        return json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

    print()
    print("=" * 72)
    print(
        f"GOOGLE SHEET RAW: {source.title}"
    )
    print("=" * 72)

    print("Downloading full XLSX...")
    xlsx_data, xlsx_headers = (
        fetch_bytes(
            source.xlsx_url
        )
    )

    if looks_like_html(
        xlsx_data,
        xlsx_headers,
    ):
        raise RuntimeError(
            f"{source.title}: вместо XLSX "
            "получен HTML. Проверь доступ."
        )

    xlsx_path.write_bytes(
        xlsx_data
    )

    print(
        f"XLSX: {len(xlsx_data):,} bytes"
    )

    print(
        f"Downloading selected gid "
        f"{source.gid} as CSV..."
    )

    (
        csv_data,
        csv_headers,
        csv_effective_url,
    ) = fetch_csv_with_fallback(
        source
    )

    csv_path.write_bytes(
        csv_data
    )

    csv_text = decode_csv(
        csv_data
    )

    rows_raw = parse_csv_rows(
        csv_text
    )
    rows, width = normalize_width(
        rows_raw
    )

    write_rows_jsonl(
        rows_path,
        rows,
    )

    preview = make_preview(
        rows,
        max_rows=20,
    )

    preview_path.write_text(
        json.dumps(
            {
                "source_key":
                    source.key,
                "title":
                    source.title,
                "gid":
                    source.gid,
                "row_count":
                    len(rows),
                "column_count":
                    width,
                "rows":
                    preview,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    nonblank_rows = sum(
        not row_is_blank(row)
        for row in rows
    )

    downloaded_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    metadata = {
        "schema_version": 1,
        "key": source.key,
        "title": source.title,
        "source_type":
            source.source_type,
        "authority":
            source.authority,
        "spreadsheet_id":
            source.spreadsheet_id,
        "gid":
            source.gid,
        "source_url":
            source.edit_url,
        "downloaded_at":
            downloaded_at,
        "raw": {
            "xlsx": {
                "path": str(xlsx_path),
                "bytes":
                    len(xlsx_data),
                "sha256":
                    sha256_bytes(
                        xlsx_data
                    ),
                "content_type":
                    xlsx_headers.get(
                        "content-type"
                    ),
            },
            "csv": {
                "path": str(csv_path),
                "effective_url":
                    csv_effective_url,
                "bytes":
                    len(csv_data),
                "sha256":
                    sha256_bytes(
                        csv_data
                    ),
                "content_type":
                    csv_headers.get(
                        "content-type"
                    ),
            },
            "rows_jsonl": {
                "path": str(rows_path),
            },
            "preview": {
                "path":
                    str(preview_path),
            },
        },
        "shape": {
            "row_count":
                len(rows),
            "nonblank_rows":
                nonblank_rows,
            "column_count":
                width,
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Rows: {len(rows):,} "
        f"(nonblank {nonblank_rows:,})"
    )
    print(
        f"Columns: {width}"
    )
    print(
        f"Saved: {source_dir}"
    )

    return metadata


def collect_all(
    *,
    force: bool = False,
) -> dict[str, Any]:
    RAW_BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    items = []

    for source in SOURCES:
        items.append(
            collect_source(
                source,
                force=force,
            )
        )

    summary = {
        "schema_version": 1,
        "sources": items,
        "totals": {
            "sources":
                len(items),
            "rows": sum(
                int(
                    item.get(
                        "shape",
                        {},
                    ).get(
                        "row_count",
                        0,
                    )
                )
                for item in items
            ),
            "nonblank_rows": sum(
                int(
                    item.get(
                        "shape",
                        {},
                    ).get(
                        "nonblank_rows",
                        0,
                    )
                )
                for item in items
            ),
        },
    }

    summary_path = (
        RAW_BASE
        / "google_sheets_sources_summary.json"
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
    print("GOOGLE SHEETS RAW DONE")
    print("=" * 72)
    print(
        f"Sources: {summary['totals']['sources']}"
    )
    print(
        f"Rows:    {summary['totals']['rows']}"
    )
    print(
        f"Nonblank:{summary['totals']['nonblank_rows']}"
    )
    print(
        f"Summary: {summary_path}"
    )

    return summary
