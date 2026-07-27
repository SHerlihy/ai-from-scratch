import unittest

from main import SimpleTokenizerV2, vocab


class SimpleTokenizerTests(unittest.TestCase):
    def test_all_known_chars(self):
        expected = [56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]

        tokenizer = SimpleTokenizerV2(vocab)
        text = """It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
        ids = tokenizer.encode(text)

        self.assertEqual(
                ids,
                expected
            )


if __name__ == "__main__":
    unittest.main()
