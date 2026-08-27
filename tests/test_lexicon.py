import unittest

from crossword.lexicon import _load_system_words, is_valid_entry


class TestIsValidEntry(unittest.TestCase):
    def test_system_dictionary_is_available(self):
        loaded = _load_system_words()
        self.assertTrue(
            loaded,
            "no system word list found. On Debian/Ubuntu: apt install wamerican",
        )

    def test_missing_system_dictionary_is_empty(self):
        self.assertEqual(_load_system_words("/nonexistent/words"), set())

    def test_real_words_pass(self):
        for word in ("LINE", "CAT", "OATEN", "GALA", "AROW"):
            self.assertTrue(is_valid_entry(word), word)

    def test_nonsense_fails(self):
        for word in ("LFA", "ETNT", "XYZ", "QXYZ"):
            self.assertFalse(is_valid_entry(word), word)

    def test_abbreviations_without_vowels_pass(self):
        self.assertTrue(is_valid_entry("GDP"))
        self.assertTrue(is_valid_entry("NFL"))

    def test_compounds_pass(self):
        self.assertTrue(is_valid_entry("BEERBAR"))
        self.assertTrue(is_valid_entry("SCROLLSAW"))

    def test_plurals_of_known_stems_pass(self):
        self.assertTrue(is_valid_entry("TESTS"))
        self.assertTrue(is_valid_entry("CARS"))

    def test_model_proposal_is_allowed_even_if_obscure(self):
        self.assertTrue(is_valid_entry("RAE", proposed=True))
        self.assertTrue(is_valid_entry("DICKCHENEY", proposed=True))

    def test_rejects_non_alpha(self):
        self.assertFalse(is_valid_entry(""))
        self.assertFalse(is_valid_entry("A"))
