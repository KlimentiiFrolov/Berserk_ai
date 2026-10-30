"""Оркестрация: load → normalize → chunk → (embed → index). Тонкий слой без собственной логики разбора."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .chunking.faq import chunk_faq
from .chunking.rules import chunk_rules
from .config import BUILD_DIR, FAQ_SHEETS, SNAPSHOT_DIR, SYMBOLS_FILE, FaqSheet
from .loaders.faq_sheets import load_faq_sheet, refresh_snapshots, snapshot_csv_path
from .loaders.rules_docx import load_rules_docx
from .normalize.faq import normalize_faq
from .normalize.rules import normalize_rules
from .schema import Chunk
from .symbols import load_symbol_names


@dataclass
class BuildResult:
    chunks: list[Chunk] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)      # что именно было прочитано (путь, sha256, дата)
    reports: dict[str, dict] = field(default_factory=dict)


def find_rules_docx(explicit: Path | None = None, snapshot_dir: Path = SNAPSHOT_DIR) -> Path:
    if explicit:
        return Path(explicit)
    found = sorted(snapshot_dir.glob("rules*.docx"))
    if len(found) != 1:
        raise SystemExit(
            f"Ожидался ровно один файл {snapshot_dir}/rules*.docx, найдено: {len(found)}. "
            "Положите туда docx правил или укажите --rules-docx."
        )
    return found[0]


def build_faq(sheet: FaqSheet, *, snapshot_dir: Path = SNAPSHOT_DIR, csv_path: Path | None = None,
              refresh: bool = False) -> BuildResult:
    raw = load_faq_sheet(sheet, snapshot_dir, refresh=refresh, csv_path=csv_path)
    records, report = normalize_faq(raw.rows, sheet.set_name)
    chunks = chunk_faq(records, doc_id=f"faq_{sheet.key}", snapshot_date=raw.snapshot_date)
    return BuildResult(
        chunks=chunks,
        inputs=[{"kind": "faq", "set_name": sheet.set_name, "gid": sheet.gid, "path": str(raw.path),
                 "sha256": raw.sha256, "snapshot_date": raw.snapshot_date}],
        reports={f"faq_{sheet.key}": asdict(report)},
    )


def build_rules(docx_path: Path, *, symbols_file: Path = SYMBOLS_FILE) -> tuple[BuildResult, object]:
    raw = load_rules_docx(docx_path, load_symbol_names(symbols_file))
    items, report = normalize_rules(raw)
    chunks = chunk_rules(items, doc_id=f"rules_{raw.version}", version=raw.version, snapshot_date=raw.snapshot_date)
    return BuildResult(
        chunks=chunks,
        inputs=[{"kind": "rules", "version": raw.version, "path": str(raw.path), "sha256": raw.sha256,
                 "snapshot_date": raw.snapshot_date}],
        reports={"rules": asdict(report)},
    ), raw


def build(sources: list[str], *, faq_sheets: list[FaqSheet] | None = None, csv_path: Path | None = None,
          rules_docx: Path | None = None, snapshot_dir: Path = SNAPSHOT_DIR, refresh: bool = False,
          symbols_file: Path = SYMBOLS_FILE) -> tuple[BuildResult, object | None]:
    total = BuildResult()
    raw_rules = None
    if "faq" in sources:
        if csv_path and not faq_sheets:
            raise ValueError("csv_path относится к одной вкладке: укажите её явно (faq_sheets=[...] / --faq-key)")
        sheets = faq_sheets or list(FAQ_SHEETS.values())
        if not csv_path:   # одна загрузка таблицы на все вкладки, а не по загрузке на каждую
            refresh_snapshots([s for s in sheets if refresh or not snapshot_csv_path(s, snapshot_dir).exists()],
                              snapshot_dir)
        for sheet in sheets:
            part = build_faq(sheet, snapshot_dir=snapshot_dir, csv_path=csv_path, refresh=False)
            total.chunks += part.chunks
            total.inputs += part.inputs
            total.reports.update(part.reports)
    if "rules" in sources:
        part, raw_rules = build_rules(find_rules_docx(rules_docx, snapshot_dir), symbols_file=symbols_file)
        total.chunks += part.chunks
        total.inputs += part.inputs
        total.reports.update(part.reports)

    ids = [c.chunk_id for c in total.chunks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise RuntimeError(f"Дубли chunk_id ({len(dupes)}): {dupes[:5]}")
    return total, raw_rules


# ---------- артефакты прогона ----------
def chunks_jsonl(chunks: list[Chunk]) -> str:
    """Канонический вид: одинаковые входные данные дают побайтно одинаковый результат."""
    return "".join(json.dumps(c.model_dump(exclude_none=True), ensure_ascii=False, sort_keys=True) + "\n"
                   for c in chunks)


def write_artifacts(result: BuildResult, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body = chunks_jsonl(result.chunks)
    (out_dir / "chunks.jsonl").write_text(body, encoding="utf-8")
    (out_dir / "reports.json").write_text(json.dumps(result.reports, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"chunks_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "chunks_path": str(out_dir / "chunks.jsonl")}


def summarize(chunks: list[Chunk]) -> dict:
    out: dict[str, dict] = {}
    for c in chunks:
        bucket = out.setdefault(c.metadata.source_type, {"total": 0, "official": 0, "community": 0})
        bucket["total"] += 1
        bucket[c.metadata.priority] += 1
    return out


def write_manifest(out_dir: Path, *, result: BuildResult, artifacts: dict, embedder_info: dict | None = None,
                   collection: str | None = None, points: int | None = None, smoke: list[dict] | None = None) -> Path:
    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "python": platform.python_version(),
        "platform": sys.platform,
        "inputs": result.inputs,
        "chunks": {"count": len(result.chunks), "by_source": summarize(result.chunks), **artifacts},
        "embedder": embedder_info,
        "collection": collection,
        "points_in_collection": points,
        "smoke": smoke,
    }
    path = Path(out_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------- индексация и smoke-тест ----------
def run_smoke(indexer, collection: str, embedder, queries: list[dict]) -> list[dict]:
    """Каждый запрос: ожидаемый чанк должен попасть в топ-3. Пустой список = тесты не заданы."""
    results = []
    for q in queries:
        hits = indexer.search(collection, embedder, q["query"], limit=3,
                              source_type=q.get("source_type"))
        expect = q["expect"]
        ok = any(all(str(h.get(k)) == str(v) for k, v in expect.items()) for h in hits)
        results.append({"query": q["query"], "expect": expect, "ok": ok,
                        "top": [{k: h.get(k) for k in ("score", "card_name", "section_id")} for h in hits]})
    return results


def reindex(result: BuildResult, *, indexer, embedder, collection: str | None, recreate: bool,
            out_dir: Path, smoke_queries: list[dict]) -> dict:
    artifacts = write_artifacts(result, out_dir)
    name = indexer.load_data(result.chunks, embedder=embedder, collection_name=collection, recreate=recreate)
    points = indexer.count(name)
    smoke = run_smoke(indexer, name, embedder, smoke_queries) if smoke_queries else []
    manifest_path = write_manifest(out_dir, result=result, artifacts=artifacts, embedder_info=embedder.info(),
                                   collection=name, points=points, smoke=smoke)
    return {"collection": name, "points": points, "expected": len(result.chunks),
            "count_ok": points >= len(result.chunks) if not recreate else points == len(result.chunks),
            "smoke": smoke, "manifest": str(manifest_path), **artifacts}
