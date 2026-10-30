import unittest

from berserk_ingest.loaders.rules_docx import NumLevel
from berserk_ingest.normalize.numbering import NumberingState, format_counter

DEFS = {
    1: {0: NumLevel("decimal", "105.%1."), 1: NumLevel("decimal", "105.%1.%2."),
        2: NumLevel("decimal", "%1.%2.%3.")},
    2: {0: NumLevel("decimal", "101.%1."), 1: NumLevel("lowerLetter", "%2."), 2: NumLevel("lowerRoman", "%3.")},
    3: {0: NumLevel("bullet", "●"), 1: NumLevel("bullet", "○")},
    4: {0: NumLevel("decimal", "9.%1.", start=5)},
}


def run(state, seq):
    return [state.advance(n, l).text for n, l in seq]


class NumberingTests(unittest.TestCase):
    def test_nested_numbers_and_reset(self):
        s = NumberingState(DEFS)
        got = run(s, [(1, 0), (1, 1), (1, 1), (1, 0), (1, 1), (1, 1), (1, 2)])
        self.assertEqual(got, ["105.1", "105.1.1", "105.1.2", "105.2", "105.2.1", "105.2.2", "2.2.1"])
        # 3-й уровень имеет шаблон «%1.%2.%3.» без префикса раздела — Word рисует его так же;
        # итоговый id пункта достраивает normalize/rules.py (см. test_rules_structure)

    def test_deeper_levels_restart_after_parent_step(self):
        s = NumberingState(DEFS)
        run(s, [(1, 0), (1, 1), (1, 1)])
        self.assertEqual(run(s, [(1, 0), (1, 1)]), ["105.2", "105.2.1"])

    def test_letters_and_roman_are_not_full(self):
        s = NumberingState(DEFS)
        s.advance(2, 0)
        a = s.advance(2, 1)
        b = s.advance(2, 1)
        self.assertEqual((a.text, a.is_full, b.text), ("a", False, "b"))
        self.assertEqual(s.advance(2, 2).text, "i")

    def test_bullets_have_no_label(self):
        lab = NumberingState(DEFS).advance(3, 0)
        self.assertTrue(lab.is_bullet)

    def test_start_value_and_unknown_list(self):
        s = NumberingState(DEFS)
        self.assertEqual(s.advance(4, 0).text, "9.5")
        self.assertIsNone(s.advance(99, 0))

    def test_lists_count_independently(self):
        s = NumberingState(DEFS)
        run(s, [(1, 0), (1, 0)])
        self.assertEqual(s.advance(2, 0).text, "101.1")

    def test_format_counter(self):
        self.assertEqual([format_counter("lowerLetter", n) for n in (1, 26, 27)], ["a", "z", "aa"])
        self.assertEqual(format_counter("upperRoman", 14), "XIV")


if __name__ == "__main__":
    unittest.main()
