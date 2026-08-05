import unittest

from bim_agents.tools import _level_matches


class LevelMatchingTests(unittest.TestCase):
    def test_matches_ordinal_and_numeric_level(self):
        self.assertTrue(_level_matches("seventh floor", "Level 7"))
        self.assertTrue(_level_matches("7th floor", "Floor 07"))

    def test_does_not_match_different_or_missing_level(self):
        self.assertFalse(_level_matches("seventh floor", "Level 17"))
        self.assertFalse(_level_matches("seventh floor", None))

    def test_matches_cross_language_contract_alias(self):
        aliases = {"ground": ["ground floor", "קרקע", "קרקע מפלס 1.5"]}
        self.assertTrue(_level_matches("ground floor", "קרקע", aliases))
        self.assertTrue(_level_matches("ground floor", "\u200fקרקע מפלס 1.5", aliases))


if __name__ == "__main__":
    unittest.main()
