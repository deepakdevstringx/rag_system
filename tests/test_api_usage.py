"""Tests for exposing model token usage through the chat SSE endpoint."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from pdf_rag.api import app


class ChatUsageEndpointTests(unittest.TestCase):
    """Ensure the chat endpoint sends per-call token counts to the client."""

    def test_chat_stream_includes_usage_event(self):
        """Include model usage records alongside answer and source events."""
        model_usage = [
            {
                "provider": "claude",
                "model": "claude-sonnet-4-5",
                "operation": "answer_draft",
                "available": True,
                "input_tokens": 120,
                "output_tokens": 9,
                "total_tokens": 129,
            }
        ]
        with patch(
            "pdf_rag.api._answer_request_with_usage",
            return_value=("A grounded answer.", ["[policy.pdf - v1 Page 1]"], model_usage),
        ):
            response = TestClient(app).post(
                "/api/chat",
                json={"message": "Ask a question", "history": []},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: usage", response.text)
        self.assertIn('"input_tokens": 120', response.text)
        self.assertIn('"output_tokens": 9', response.text)
        self.assertIn('"total_tokens": 129', response.text)
        self.assertIn("event: token", response.text)


if __name__ == "__main__":
    unittest.main()