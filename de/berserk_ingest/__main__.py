"""CLI: python -m berserk_ingest {chunks|reindex|search|symbols} ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import BUILD_DIR, DEFAULT_COLLECTION, DEFAULT_EMBEDDER, DATA_DIR, FAQ_SHEETS, FaqSheet, QDRANT_URL, SNAPSHOT_DIR, SYMBOLS_FILE
from .pipeline import build, find_rules_docx, reindex, summarize, write_artifacts, write_manifest
from .symbols import export_symbols

SMOKE_FILE = DATA_DIR / "smoke_queries.json"


def _add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", choices=["faq", "rules", "all"], default="all")
    p.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    p.add_argument("--rules-docx", type=Path, help="путь к docx правил (по умолчанию data/snapshots/rules*.docx)")
    p.add_argument("--faq-key", action="append", help=f"ключ вкладки FAQ из config ({', '.join(FAQ_SHEETS)}); можно несколько")
    p.add_argument("--faq-csv", type=Path, help="явный путь к CSV-снапшоту вкладки (только для одной вкладки)")
    p.add_argument("--gid", help="своя вкладка: gid из адресной строки (необязателен, вкладка ищется по --set-name)")
    p.add_argument("--set-name", help="своя вкладка, которой нет в config: имя точно как в таблице, например '9. Новый выпуск'")
    p.add_argument("--refresh", action="store_true", help="перекачать FAQ из Google Sheets (иначе берётся снапшот)")
    p.add_argument("--symbols-file", type=Path, default=SYMBOLS_FILE)
    p.add_argument("--out", type=Path, default=BUILD_DIR)


def _sheets(args) -> list[FaqSheet] | None:
    if args.gid or args.set_name:
        if not args.set_name:
            raise SystemExit("--gid требует --set-name")
        from .config import FAQ_SPREADSHEET_ID
        key = args.set_name.split(".")[0].strip() or args.gid
        return [FaqSheet(key, FAQ_SPREADSHEET_ID, args.gid or "", args.set_name)]
    if args.faq_csv and not args.faq_key:
        raise SystemExit(f"--faq-csv относится к одной вкладке: добавьте --faq-key ({', '.join(FAQ_SHEETS)})")
    if args.faq_key:
        unknown = [k for k in args.faq_key if k not in FAQ_SHEETS]
        if unknown:
            raise SystemExit(f"Неизвестные ключи вкладок: {unknown}; известны: {list(FAQ_SHEETS)}")
        return [FAQ_SHEETS[k] for k in args.faq_key]
    return None


def _sources(args) -> list[str]:
    return ["faq", "rules"] if args.source == "all" else [args.source]


def _print_reports(result) -> None:
    for name, rep in result.reports.items():
        if name.startswith("faq"):
            print(f"[{name}] qna={rep['qna']} rulings={rep['rulings']} без ответа={rep['unanswered']} "
                  f"approved={rep['approved']} not_approved={rep['not_approved']} unknown={rep['unknown']}")
        else:
            print(f"[rules] пунктов={rep['items']} {rep['by_kind']} коллизий id={rep['id_collisions']} "
                  f"ссылки {rep['refs_resolved']}/{rep['refs_total']} символов без имени={rep['symbols_unnamed']}/{rep['symbols_total']}")
            if rep["refs_unresolved"]:
                print("        неразрешённые ссылки:", ", ".join(rep["refs_unresolved"][:6]))
        for w in rep.get("warnings", []):
            print(f"        ! {w}")
    print("чанков:", json.dumps(summarize(result.chunks), ensure_ascii=False))


def cmd_chunks(args) -> int:
    result, _ = build(_sources(args), faq_sheets=_sheets(args), csv_path=args.faq_csv, rules_docx=args.rules_docx,
                      snapshot_dir=args.snapshot_dir, refresh=args.refresh, symbols_file=args.symbols_file)
    artifacts = write_artifacts(result, args.out)
    write_manifest(args.out, result=result, artifacts=artifacts)
    _print_reports(result)
    print(f"записано: {artifacts['chunks_path']}  sha256={artifacts['chunks_sha256'][:16]}…")
    return 0


def _client(args):
    from qdrant_client import QdrantClient
    if args.local == ":memory:":
        return QdrantClient(":memory:")
    if args.local:
        return QdrantClient(path=args.local)
    return QdrantClient(url=args.qdrant_url)


def _embedder(args):
    """--dense-model задаёт модели FastEmbed по имени; иначе берётся пресет --embedder."""
    from .embedders import custom_embedder, get_embedder
    if args.dense_model:
        return custom_embedder(args.dense_model, None if args.no_sparse else args.sparse_model)
    return get_embedder(args.embedder)


def cmd_reindex(args) -> int:
    from .indexer import Indexer
    result, _ = build(_sources(args), faq_sheets=_sheets(args), csv_path=args.faq_csv, rules_docx=args.rules_docx,
                      snapshot_dir=args.snapshot_dir, refresh=args.refresh, symbols_file=args.symbols_file)
    _print_reports(result)
    smoke_queries = json.loads(SMOKE_FILE.read_text(encoding="utf-8")) if SMOKE_FILE.exists() else []
    present = {c.metadata.source_type for c in result.chunks}
    smoke_queries = [q for q in smoke_queries if q.get("source_type") in present]

    embedder = _embedder(args)
    summary = reindex(result, indexer=Indexer(_client(args)), embedder=embedder,
                      collection=args.collection or DEFAULT_COLLECTION or None,
                      recreate=not args.no_recreate, out_dir=args.out, smoke_queries=smoke_queries)
    print(f"коллекция {summary['collection']}: точек {summary['points']} (ожидалось {summary['expected']}) "
          f"-> {'OK' if summary['count_ok'] else 'РАСХОЖДЕНИЕ'}")
    failed = [s for s in summary["smoke"] if not s["ok"]]
    for s in summary["smoke"]:
        print(f"  smoke {'OK ' if s['ok'] else 'FAIL'} {s['query']!r} -> {s['top']}")
    print("манифест:", summary["manifest"])
    if not summary["count_ok"] or (args.strict and failed):
        return 1
    return 0


def cmd_search(args) -> int:
    from .indexer import Indexer, default_collection_name
    embedder = _embedder(args)
    name = args.collection or DEFAULT_COLLECTION or default_collection_name(embedder)
    hits = Indexer(_client(args)).search(name, embedder, args.query, limit=args.limit, mode=args.mode,
                                         source_type=args.source_type, priority=args.priority)
    for h in hits:
        label = h.get("card_name") or h.get("section_id")
        print(f"{h['score']:.4f}  [{h['source_type']}/{h['priority']}] {label}\n    {h['text'][:200]!r}")
    return 0


def cmd_symbols(args) -> int:
    from .loaders.rules_docx import load_rules_docx
    from .symbols import load_symbol_names
    raw = load_rules_docx(find_rules_docx(args.rules_docx, args.snapshot_dir), load_symbol_names(args.symbols_file))
    tpl = export_symbols(raw.symbols, args.out / "symbols")
    unnamed = sum(1 for s in raw.symbols.values() if not s.name)
    print(f"символов: {len(raw.symbols)}, без имени: {unnamed}. Заполните имена в {tpl} и сохраните как {args.symbols_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="berserk_ingest", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("chunks", help="собрать чанки без эмбеддингов и Qdrant (быстро, для проверки)")
    _add_source_args(p)
    p.set_defaults(func=cmd_chunks)

    def qdrant_args(p):
        p.add_argument("--embedder", default=DEFAULT_EMBEDDER, help="пресет из embedders.PRESETS")
        p.add_argument("--dense-model", help="имя dense-модели FastEmbed (вместо пресета), напр. intfloat/multilingual-e5-large")
        p.add_argument("--sparse-model", default="Qdrant/bm25", help="sparse-модель к --dense-model (по умолчанию Qdrant/bm25)")
        p.add_argument("--no-sparse", action="store_true", help="с --dense-model: только dense, без BM25")
        p.add_argument("--collection")
        p.add_argument("--qdrant-url", default=QDRANT_URL)
        p.add_argument("--local", help="встроенный Qdrant без сервера: ':memory:' или путь к папке")

    p = sub.add_parser("reindex", help="чанки -> эмбеддинги -> Qdrant + манифест + smoke-тест")
    _add_source_args(p)
    qdrant_args(p)
    p.add_argument("--no-recreate", action="store_true", help="не пересоздавать коллекцию (синхронизация по doc_id)")
    p.add_argument("--strict", action="store_true", help="падать (exit 1), если smoke-тест не прошёл")
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("search", help="поисковый запрос к коллекции (ручная проверка)")
    p.add_argument("query")
    qdrant_args(p)
    p.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--source-type", choices=["faq", "rules"])
    p.add_argument("--priority", choices=["official", "community"])
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("symbols", help="выгрузить картинки-символы и шаблон для имён")
    p.add_argument("--rules-docx", type=Path)
    p.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    p.add_argument("--symbols-file", type=Path, default=SYMBOLS_FILE)
    p.add_argument("--out", type=Path, default=BUILD_DIR)
    p.set_defaults(func=cmd_symbols)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ImportError as exc:
        print(f"Не установлена зависимость {exc.name!r}. Выполните: pip install -r requirements.txt "
              "(для e5-base ещё и requirements-optional.txt)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
