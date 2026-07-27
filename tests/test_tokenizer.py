import unittest

from main import SimpleTokenizerV2, vocab


class SimpleTokenizerTests(unittest.TestCase):
    def encode_all_known_chars(self, tokenizer):
        text = """It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
        ids = tokenizer.encode(text)
        return ids

    def encode_endoftext_char(self, tokenizer):
        text = """
            Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.
            """
        ids = tokenizer.encode(text)
        return ids

    def test_all_known_chars_encoded(self):
        expected = [56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]

        tokenizer = SimpleTokenizerV2(vocab)
        encoded = self.encode_all_known_chars(tokenizer)

        self.assertEqual(
                encoded,
                expected
            )

    def test_all_known_chars_decoded(self):
        expected = """It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""

        tokenizer = SimpleTokenizerV2(vocab)
        encoded = self.encode_all_known_chars(tokenizer)

        decoded = tokenizer.decode(encoded)

        self.assertEqual(
                decoded,
                expected
            )

    def test_endoftext_char(self):
        expected = [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]

        tokenizer = SimpleTokenizerV2(vocab)
        encoded = self.encode_endoftext_char(tokenizer)

        self.assertEqual(
                encoded,
                expected
            )



if __name__ == "__main__":
    unittest.main()
