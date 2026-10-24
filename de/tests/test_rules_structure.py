import unittest
from pathlib import Path

from berserk_ingest.chunking.rules import chunk_rules
from berserk_ingest.loaders.rules_docx import NumLevel, RawParagraph, RawRules
from berserk_ingest.normalize.rules import normalize_rules

NUMBERING = {
    1: {0: NumLevel("decimal", "105.%1."), 1: NumLevel("decimal", "105.%1.%2."),
        2: NumLevel("decimal", "%1.%2.%3.")},
    2: {0: NumLevel("lowerLetter", "105.%1.")},           # самостоятельный список a), b)
    3: {0: NumLevel("bullet", "●"), 1: NumLevel("bullet", "○")},
    4: {0: NumLevel("decimal", "106.%1.")},
}


def make(paragraphs):
    ps = [RawParagraph(i, p[0], p[1], p[2], *(p[3:] + (None, None))[:2]) for i, p in enumerate(paragraphs)]
    return RawRules(Path("x.docx"), "0", "t", "8.0-1", "2026-09-04", ps, NUMBERING)


H1 = ("p", "Heading 1", "Часть 1. Основы.")
H2_105 = ("p", "Heading 2", "105. Базовые свойства.")
H2_106 = ("p", "Heading 2", "106. Небазовые свойства.")


class RulesStructureTests(unittest.TestCase):
    def test_front_matter_is_ignored(self):
        raw = make([("p", "Heading 1", "Разработчики."), ("p", "normal", "Иванов"), H1, H2_105,
                    ("p", "normal", "Пункт.", 1, 0)])
        items, _ = normalize_rules(raw)
        self.assertEqual([i.point_id for i in items], ["105.1"])

    def test_hierarchy_ids_and_parents(self):
        raw = make([H1, H2_105, ("p", "normal", "Первый.", 1, 0), ("p", "normal", "Вложенный.", 1, 1),
                    ("p", "normal", "Второй.", 1, 0)])
        items, _ = normalize_rules(raw)
        self.assertEqual([(i.point_id, i.parent_id) for i in items],
                         [("105.1", None), ("105.1.1", "105.1"), ("105.2", None)])
        self.assertEqual(items[1].path_ids, ["105.1", "105.1.1"])

    def test_deep_level_without_section_prefix_gets_parent_based_id(self):
        raw = make([H1, H2_105, ("p", "normal", "А.", 1, 0), ("p", "normal", "Б.", 1, 1), ("p", "normal", "В.", 1, 2)])
        items, rep = normalize_rules(raw)
        self.assertEqual([i.point_id for i in items], ["105.1", "105.1.1", "105.1.1.1"])
        self.assertEqual(items[2].parent_id, "105.1.1")
        self.assertEqual(rep.section_mismatches, [])

    def test_bullets_and_plain_paragraphs_join_previous_point(self):
        raw = make([H1, H2_105, ("p", "normal", "Для игры нужны:", 1, 0), ("p", "normal", "карты;", 3, 1),
                    ("p", "normal", "кубик.", 3, 1), ("p", "normal", "Продолжение мысли.")])
        items, _ = normalize_rules(raw)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].lines, ["Для игры нужны:", "– карты;", "– кубик.", "Продолжение мысли."])

    def test_letter_list_attaches_to_last_decimal_point(self):
        raw = make([H1, H2_105, ("p", "normal", "Пункт.", 1, 0), ("p", "normal", "Второй.", 1, 0),
                    ("p", "normal", "первое;", 2, 0), ("p", "normal", "второе.", 2, 0)])
        items, rep = normalize_rules(raw)
        ids = [i.point_id for i in items]
        self.assertEqual(ids, ["105.1", "105.2", "105.2.a", "105.2.b"])
        self.assertEqual(items[2].parent_id, "105.2")
        self.assertEqual(rep.id_collisions, 0)

    def test_empty_heading_does_not_reset_section(self):
        raw = make([H1, H2_105, ("p", "normal", "А.", 1, 0), ("p", "Heading 2", ""), ("p", "normal", "Б.", 1, 0)])
        items, rep = normalize_rules(raw)
        self.assertEqual({i.section_id for i in items}, {"105"})
        self.assertEqual(rep.section_mismatches, [])

    def test_terms_part_makes_one_item_per_bullet(self):
        raw = make([("p", "Heading 1", "Часть 7. Терминология."),
                    ("p", "normal", "Аккениец. Класс карты.", 3, 0),
                    ("p", "normal", "Атака. Понятие, включающее:", 3, 0),
                    ("p", "normal", "удар;", 3, 1)])
        items, _ = normalize_rules(raw)
        self.assertEqual([i.kind for i in items], ["term", "term"])
        self.assertEqual(items[1].lines, ["Атака. Понятие, включающее:", "– удар;"])
        chunks = chunk_rules(items, doc_id="rules_8.0-1", version="8.0-1", snapshot_date="d")
        self.assertEqual(chunks[0].metadata.section_path, "Часть 7 > 7.аккениец")

    def test_table_becomes_own_item_with_caption(self):
        raw = make([H1, H2_105, ("p", "normal", "Пункт.", 1, 0), ("p", "normal", "Таблица способностей."),
                    ("table", "table", "Ближняя | Магическая: удар")])
        items, _ = normalize_rules(raw)
        self.assertEqual([i.kind for i in items], ["point", "table"])
        self.assertEqual(items[1].lines[0], "Таблица способностей")
        self.assertEqual(items[0].lines, ["Пункт."])        # подпись не осталась в тексте пункта

    def test_cross_references_are_resolved(self):
        raw = make([H1, H2_105, ("p", "normal", "Пункт.", 1, 0), ("p", "normal", "См. п. 105.1 и пп. 105.7.", 1, 0)])
        _, rep = normalize_rules(raw)
        self.assertEqual((rep.refs_total, rep.refs_resolved), (2, 1))
        self.assertEqual(len(rep.refs_unresolved), 1)

    def test_section_mismatch_is_reported(self):
        raw = make([H1, H2_105, ("p", "normal", "Не тот раздел.", 4, 0)])   # список 106.* внутри раздела 105
        _, rep = normalize_rules(raw)
        self.assertEqual(len(rep.section_mismatches), 1)


class RulesChunkTests(unittest.TestCase):
    def chunks(self, paragraphs, **kw):
        items, _ = normalize_rules(make(paragraphs))
        return chunk_rules(items, doc_id="rules_8.0-1", version="8.0-1", snapshot_date="2026-09-04", **kw)

    def test_child_chunk_carries_parent_context_and_ids(self):
        ch = self.chunks([H1, H2_105, ("p", "normal", "Элитность. Принадлежность карты.", 1, 0),
                          ("p", "normal", "Если стоимость X, то определить.", 1, 1)])
        child = ch[1]
        self.assertIn("Родительский пункт 105.1: Элитность. Принадлежность карты.", child.text)
        self.assertIn("105.1.1. Если стоимость X, то определить.", child.text)
        self.assertEqual(child.metadata.section_path, "Часть 1 > 105. Базовые свойства > 105.1 > 105.1.1")
        self.assertEqual((child.metadata.chunk_id, child.metadata.priority), ("rules_8.0-1_105.1.1", "official"))

    def test_long_point_is_split_with_same_header(self):
        long = " ".join(f"Предложение номер {i} про игровые правила." for i in range(120))
        ch = self.chunks([H1, H2_105, ("p", "normal", long, 1, 0)], max_chars=800)
        self.assertGreater(len(ch), 1)
        self.assertEqual(ch[0].metadata.chunk_id, "rules_8.0-1_105.1#1")
        self.assertTrue(all(len(c.text) <= 900 for c in ch))
        self.assertTrue(all(c.text.startswith("Правила ККИ «Берсерк» 8.0-1") for c in ch))
        self.assertEqual(len({c.chunk_id for c in ch}), len(ch))

    def test_table_split_keeps_whole_rows(self):
        rows = "\n".join(f"Ряд {i} | Колонка: значение {i}; Другая: {i * 7}" for i in range(60))
        ch = self.chunks([H1, H2_105, ("p", "normal", "Пункт.", 1, 0), ("p", "normal", "Таблица ударов."),
                          ("table", "table", rows)], max_chars=700)
        table_chunks = [c for c in ch if ".t" in c.metadata.section_id]
        self.assertGreater(len(table_chunks), 1)
        for c in table_chunks:
            for line in c.text.splitlines():
                if line.startswith("Ряд "):
                    self.assertRegex(line, r"^Ряд \d+ \| Колонка: значение \d+; Другая: \d+$")
            self.assertIn("Таблица ударов", c.text)


REAL = next(iter(sorted((Path(__file__).parent.parent / "data" / "snapshots").glob("rules*.docx"))), None)
import os
REAL = Path(os.environ["BERSERK_RULES_DOCX"]) if os.environ.get("BERSERK_RULES_DOCX") else REAL


@unittest.skipUnless(REAL and REAL.exists(), "нужен docx правил: data/snapshots/rules*.docx или BERSERK_RULES_DOCX")
class RealRulebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from berserk_ingest.loaders.rules_docx import load_rules_docx
        cls.raw = load_rules_docx(REAL)
        cls.items, cls.report = normalize_rules(cls.raw)
        cls.chunks = chunk_rules(cls.items, doc_id=f"rules_{cls.raw.version}", version=cls.raw.version,
                                 snapshot_date=cls.raw.snapshot_date)

    def test_version_and_date_from_title(self):
        self.assertEqual((self.raw.version, self.raw.snapshot_date), ("8.0-1", "2026-09-04"))

    def test_numbering_is_consistent(self):
        self.assertEqual(self.report.id_collisions, 0)
        self.assertEqual(self.report.section_mismatches, [])

    def test_cross_references_resolve(self):
        # Номера восстановлены правильно, если на них указывают ссылки из самого текста правил.
        ratio = self.report.refs_resolved / self.report.refs_total
        self.assertGreaterEqual(ratio, 0.97, self.report.refs_unresolved)

    def test_known_points(self):
        by = {i.point_id: i for i in self.items}
        self.assertTrue(by["105.2.3"].text.startswith("Если стоимость карты имеет переменную величину"))
        self.assertEqual(by["105.2.3"].parent_id, "105.2")
        self.assertEqual(by["306.3.b"].parent_id, "306.3")

    def test_chunks_valid(self):
        ids = [c.chunk_id for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(max(len(c.text) for c in self.chunks), 1700)
        self.assertGreater(len(self.chunks), 1000)

    def test_symbols_are_marked_in_text(self):
        self.assertTrue(any("[символ]" in i.text for i in self.items))


if __name__ == "__main__":
    unittest.main()
