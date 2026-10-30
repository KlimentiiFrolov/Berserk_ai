import unittest
from pathlib import Path

from pydantic import ValidationError

from berserk_ingest.chunking.faq import chunk_faq
from berserk_ingest.loaders.faq_sheets import parse_csv
from berserk_ingest.normalize.faq import normalize_faq
from berserk_ingest.schema import Chunk, ChunkMetadata

FIXTURE = Path(__file__).parent / "fixtures" / "faq_4_sample.csv"
SET = "4. Ложные боги"


def sample():
    rows = parse_csv(FIXTURE.read_text(encoding="utf-8"))
    return normalize_faq(rows, SET)


class FaqNormalizeTests(unittest.TestCase):
    def test_counts_and_report(self):
        records, rep = sample()
        self.assertEqual((rep.qna, rep.rulings, rep.unanswered), (13, 8, 0))
        self.assertEqual(len(records), 21)
        self.assertEqual(rep.cards_without_content, 4)      # Авиал, Мастер осад, Вейла, Орк-маг
        self.assertEqual(rep.header_row, 1)

    def test_preamble_is_not_content(self):
        records, _ = sample()
        self.assertFalse(any("Это FAQ" in (r.card_name or "") for r in records))

    def test_card_block_spreads_over_rows(self):
        records, _ = sample()
        mirage = [r for r in records if r.card_name == "Мираж"]
        self.assertEqual([r.kind for r in mirage], ["qna", "ruling"])    # пустой «Вопрос» = ruling
        self.assertTrue(all(r.card_number == 3 for r in mirage))
        tugard = [r for r in records if r.card_name == "Стражник Тугарда"]
        self.assertEqual(len(tugard), 2)
        self.assertTrue(all(r.card_number == 12 and r.kind == "qna" for r in tugard))
        self.assertEqual(len([r for r in records if r.card_name == "Огнепоклонница"]), 3)

    def test_backslash_preserved(self):
        records, _ = sample()
        self.assertIn("ЗоЗ\\ЗоМ", next(r for r in records if r.card_name == "Жрица Дзара").question)

    def test_approval_checkbox_maps_to_priority(self):
        csv_text = ("Номер,Название,Вопрос,Ответ,Ответ утвержден\n"
                    "1,Карта А,Вопрос 1?,Да.,TRUE\n"
                    ",,Вопрос 2?,Нет.,FALSE\n"
                    ",,Вопрос 3?,Пожалуй.,\n")
        records, rep = normalize_faq(parse_csv(csv_text), SET)
        self.assertEqual([r.approval for r in records], ["approved", "not_approved", "unknown"])
        chunks = chunk_faq(records, doc_id="faq_4", snapshot_date="2026-10-10")
        self.assertEqual([c.metadata.priority for c in chunks], ["official", "community", "community"])
        self.assertEqual((rep.approved, rep.not_approved, rep.unknown), (1, 1, 1))

    def test_question_without_answer_is_skipped_and_counted(self):
        records, rep = normalize_faq(parse_csv("Номер,Название,Вопрос,Ответ,Ответ утвержден\n1,А,Вопрос?,,\n"), SET)
        self.assertEqual((len(records), rep.unanswered), (0, 1))

    def test_header_found_by_name_not_position(self):
        text = "Вопрос,Ответ,Название,Номер,Ответ утвержден\nЧто?,Это.,Карта Б,7,TRUE\n"
        records, _ = normalize_faq(parse_csv(text), SET)
        self.assertEqual((records[0].card_name, records[0].card_number, records[0].approval), ("Карта Б", 7, "approved"))

    def test_multiline_cell_is_one_row(self):
        rows = parse_csv('a,"строка 1\nстрока 2",c\nx,y,z\n')
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], "строка 1\nстрока 2")


class FaqChunkTests(unittest.TestCase):
    def test_text_contains_card_and_release(self):
        records, _ = sample()
        chunks = chunk_faq(records, doc_id="faq_4", snapshot_date="2026-10-10")
        text = next(c.text for c in chunks if c.metadata.card_name == "Оглар")
        self.assertEqual(text, "FAQ: Ложные боги\nКарта: Оглар\nОтвет: -1 от атак существ с базовым значением ЛАПКИ 2 или более")

    def test_ids_are_stable_unique_and_order_independent(self):
        records, _ = sample()
        a = chunk_faq(records, doc_id="faq_4", snapshot_date="d")
        b = chunk_faq(list(reversed(records)), doc_id="faq_4", snapshot_date="d")
        self.assertEqual(len({c.chunk_id for c in a}), len(a))
        self.assertEqual({c.chunk_id for c in a}, {c.chunk_id for c in b})

    def test_identical_records_get_distinct_ids(self):
        text = "Номер,Название,Вопрос,Ответ,Ответ утвержден\n1,А,В?,О.,\n,,В?,О.,\n"
        records, _ = normalize_faq(parse_csv(text), SET)
        ids = [c.chunk_id for c in chunk_faq(records, doc_id="faq_4", snapshot_date="d")]
        self.assertEqual(len(set(ids)), 2)


class SchemaTests(unittest.TestCase):
    def base(self, **kw):
        d = dict(chunk_id="x", doc_id="d", source_type="faq", priority="community", source_version="v",
                 snapshot_date="2026-10-10", card_name="К", set_name=SET, faq_kind="qna", approval_status="unknown")
        d.update(kw)
        return d

    def test_valid(self):
        Chunk(text="t", metadata=ChunkMetadata(**self.base()))

    def test_faq_requires_card_name(self):
        with self.assertRaises(ValidationError):
            ChunkMetadata(**self.base(card_name=None))

    def test_rules_must_be_official_with_section(self):
        with self.assertRaises(ValidationError):
            ChunkMetadata(**self.base(source_type="rules", priority="community", section_id="1", section_path="p"))
        with self.assertRaises(ValidationError):
            ChunkMetadata(**self.base(source_type="rules", priority="official"))

    def test_unknown_priority_rejected(self):
        with self.assertRaises(ValidationError):
            ChunkMetadata(**self.base(priority="Official"))


if __name__ == "__main__":
    unittest.main()
