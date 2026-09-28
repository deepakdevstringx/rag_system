"""Tests for PDF text chunking."""

import unittest

from pdf_rag.text_splitter import split_text_into_chunks


class TextSplitterTests(unittest.TestCase):
    """Verify splitting produces bounded chunks and retains source content."""

    def test_split_text_respects_chunk_limit_and_preserves_content(self):
        """Split long input into bounded chunks without dropping its text."""
        input_text = "First sentence. " + ("Useful policy detail. " * 20)
        chunks = split_text_into_chunks(input_text, chunk_size=80, overlap=10)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))
        self.assertIn("First sentence.", chunks[0])
        self.assertIn("Useful policy detail.", " ".join(chunks))


if __name__ == "__main__":
    unittest.main()