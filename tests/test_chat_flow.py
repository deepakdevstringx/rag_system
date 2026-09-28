"""Tests for conversational input routing."""

import unittest

from pdf_rag.chat import is_simple_greeting


class ChatFlowTests(unittest.TestCase):
    """Verify greetings are separated from document-search questions."""

    def test_recognizes_standalone_greetings(self):
        """Accept common greeting-only input, including punctuation."""
        for greeting in ("hi", "Hello!", "good morning"):
            with self.subTest(greeting=greeting):
                self.assertTrue(is_simple_greeting(greeting))

    def test_does_not_treat_document_questions_as_greetings(self):
        """Keep substantive document questions on the retrieval path."""
        self.assertFalse(is_simple_greeting("hi, what is the leave policy?"))
        self.assertFalse(is_simple_greeting("leave policy"))


if __name__ == "__main__":
    unittest.main()