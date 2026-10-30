"""Картинки-символы в правилах (иконки стоимости, движения, стихий и т.д.).

Без них фразы вроде «число над символом [символ]» теряют смысл. Каждая картинка получает ключ
(sha1[:10]); человекочитаемые имена хранятся в data/symbols.json: {"1bb5e580da": "кристалл", ...}.
Команда `symbols` выгружает картинки и шаблон для заполнения, дальше имена подхватываются автоматически.
"""
from __future__ import annotations

import json
from pathlib import Path

from .loaders.rules_docx import SymbolInfo


def load_symbol_names(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}


def export_symbols(symbols: dict[str, SymbolInfo], out_dir: Path) -> Path:
    """Сохраняет картинки и symbols_template.json (самые частые — первыми)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(symbols.values(), key=lambda s: (-s.count, s.key))
    template = {}
    for s in ordered:
        (out_dir / f"{s.key}.{s.ext}").write_bytes(s.blob)
        template[s.key] = s.name or ""
    tpl_path = out_dir / "symbols_template.json"
    tpl_path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [{"key": s.key, "file": f"{s.key}.{s.ext}", "count": s.count, "name": s.name, "context": s.context}
              for s in ordered]
    (out_dir / "symbols_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return tpl_path
