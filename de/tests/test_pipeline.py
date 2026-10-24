import json
import tempfile
import unittest
from pathlib import Path

from berserk_ingest import __main__ as cli
from berserk_ingest.config import FAQ_SHEETS
from berserk_ingest.loaders.faq_sheets import load_faq_sheet
from berserk_ingest.pipeline import (build, chunks_jsonl, find_rules_docx, run_smoke, summarize, write_artifacts,
                                     write_manifest)

FIX = Path(__file__).parent / "fixtures" / "faq_4_sample.csv"


def faq_build():
    return build(["faq"], faq_sheets=[FAQ_SHEETS["4"]], csv_path=FIX)[0]


class PipelineTests(unittest.TestCase):
    def test_snapshot_date_comes_from_meta_not_from_run_time(self):
        raw = load_faq_sheet(FAQ_SHEETS["4"], FIX.parent, csv_path=FIX)
        self.assertEqual(raw.snapshot_date, "2026-10-10")
        self.assertTrue(all(c.metadata.snapshot_date == "2026-10-10" for c in faq_build().chunks))

    def test_same_input_gives_byte_identical_chunks(self):
        a, b = faq_build(), faq_build()
        self.assertEqual(chunks_jsonl(a.chunks), chunks_jsonl(b.chunks))
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            self.assertEqual(write_artifacts(a, Path(d1))["chunks_sha256"], write_artifacts(b, Path(d2))["chunks_sha256"])

    def test_manifest_contents(self):
        res = faq_build()
        with tempfile.TemporaryDirectory() as d:
            art = write_artifacts(res, Path(d))
            path = write_manifest(Path(d), result=res, artifacts=art, collection="c", points=21)
            m = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(m["chunks"]["count"], 21)
        self.assertEqual(m["inputs"][0]["sha256"], res.inputs[0]["sha256"])
        self.assertEqual(m["chunks"]["by_source"]["faq"]["community"], 21)
        self.assertEqual((m["collection"], m["points_in_collection"]), ("c", 21))

    def test_missing_snapshot_with_explicit_path_fails_loudly(self):
        with self.assertRaises(FileNotFoundError):
            load_faq_sheet(FAQ_SHEETS["4"], FIX.parent, csv_path=FIX.with_name("nope.csv"))

    def test_rules_docx_must_be_unambiguous(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                find_rules_docx(None, Path(d))
            (Path(d) / "rules_a.docx").write_bytes(b"")
            (Path(d) / "rules_b.docx").write_bytes(b"")
            with self.assertRaises(SystemExit):
                find_rules_docx(None, Path(d))

    def test_summary(self):
        self.assertEqual(summarize(faq_build().chunks), {"faq": {"total": 21, "official": 0, "community": 21}})

    def test_smoke_logic(self):
        class FakeIx:
            def search(self, name, emb, q, limit, source_type=None):
                return [{"card_name": "Оглар", "score": 1.0}, {"card_name": "Мираж", "score": 0.5}]
        res = run_smoke(FakeIx(), "c", None, [
            {"query": "q1", "source_type": "faq", "expect": {"card_name": "Мираж"}},
            {"query": "q2", "source_type": "faq", "expect": {"card_name": "Нет такой"}}])
        self.assertEqual([r["ok"] for r in res], [True, False])


class CliTests(unittest.TestCase):
    def test_chunks_command_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            rc = cli.main(["chunks", "--source", "faq", "--faq-csv", str(FIX), "--out", d])
            self.assertEqual(rc, 0)
            lines = (Path(d) / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 21)
            self.assertTrue((Path(d) / "manifest.json").exists())
            self.assertTrue((Path(d) / "reports.json").exists())

    def test_custom_sheet_requires_set_name(self):
        with self.assertRaises(SystemExit):
            cli.main(["chunks", "--source", "faq", "--gid", "123"])


if __name__ == "__main__":
    unittest.main()
