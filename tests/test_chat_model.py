"""Tests for LangChain provider/model selection."""

import os
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace

from pdf_rag.chat_model import (
    collect_model_usage,
    get_chat_selection,
    invoke_chat,
    resolve_chat_settings,
    set_chat_model,
    set_chat_provider,
)


class ChatModelSelectionTests(unittest.TestCase):
    """Verify configured chat providers and model identifiers are resolved."""

    def test_resolve_ollama_model(self):
        """Keep the local Ollama model as a valid selectable provider."""
        self.assertEqual(
            resolve_chat_settings(provider="ollama", model="llama3.2"),
            ("ollama", "llama3.2"),
        )

    def test_resolve_gemini_model(self):
        """Accept a Gemini model name through the same provider interface."""
        self.assertEqual(
            resolve_chat_settings(provider="gemini", model="gemini-2.5-flash"),
            ("gemini", "gemini-2.5-flash"),
        )

    def test_resolve_claude_model(self):
        """Accept a Claude model name through the shared provider interface."""
        self.assertEqual(
            resolve_chat_settings(provider="claude", model="claude-sonnet-4-5"),
            ("claude", "claude-sonnet-4-5"),
        )

    def test_reject_unknown_provider(self):
        """Reject unsupported providers with a clear configuration error."""
        with self.assertRaisesRegex(ValueError, "ollama.*gemini"):
            resolve_chat_settings(provider="unknown")

    def test_switch_providers_during_runtime(self):
        """Switch Ollama to Gemini and back without making an API request."""
        original_provider, original_model = get_chat_selection()
        with patch.dict(
            os.environ,
            {
                "GOOGLE_API_KEY": "test-google-key-only",
                "ANTHROPIC_API_KEY": "test-anthropic-key-only",
            },
        ):
            try:
                self.assertEqual(set_chat_provider("claude")[0], "claude")
                self.assertEqual(get_chat_selection()[1], "claude-sonnet-4-5")
                self.assertEqual(set_chat_provider("gemini")[0], "gemini")
                self.assertEqual(get_chat_selection()[1], "gemini-2.5-flash")
                self.assertEqual(set_chat_provider("ollama")[0], "ollama")
            finally:
                set_chat_provider(original_provider)
                set_chat_model(original_model)

    def test_claude_invoke_failure_falls_back_to_ollama(self):
        """Use the configured local model when the primary Claude call fails."""
        failing_claude = SimpleNamespace(
            invoke=lambda _messages: (_ for _ in ()).throw(ConnectionError("offline"))
        )
        ollama_response = SimpleNamespace(content="fallback answer")
        ollama_model = SimpleNamespace(invoke=lambda _messages: ollama_response)

        with patch("pdf_rag.chat_model._active_provider", "claude"), patch(
            "pdf_rag.chat_model._active_model", "claude-sonnet-4-5"
        ), patch(
            "pdf_rag.chat_model._create_chat_model",
            side_effect=[failing_claude, ollama_model],
        ) as create_model:
            answer = invoke_chat([{"role": "user", "content": "Question"}])

        self.assertEqual(answer, "fallback answer")
        self.assertEqual(create_model.call_args_list[1].kwargs["provider"], "ollama")
        self.assertEqual(create_model.call_args_list[1].kwargs["model"], "llama3.2")

    def test_invoke_prints_model_return_to_python_terminal(self):
        """Print the actual returned text with provider and model labels."""
        response_model = SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(content="The policy grants 12 days.")
        )
        output = io.StringIO()

        with patch("pdf_rag.chat_model._active_provider", "claude"), patch(
            "pdf_rag.chat_model._active_model", "claude-sonnet-4-5"
        ), patch("pdf_rag.chat_model._create_chat_model", return_value=response_model), redirect_stdout(output):
            result = invoke_chat([{"role": "user", "content": "How many days?"}])

        self.assertEqual(result, "The policy grants 12 days.")
        logged_response = output.getvalue()
        self.assertIn("MODEL RESPONSE", logged_response)
        self.assertIn("Provider: claude", logged_response)
        self.assertIn("Model: claude-sonnet-4-5", logged_response)
        self.assertIn("The policy grants 12 days.", logged_response)

    def test_invoke_collects_provider_token_usage(self):
        """Capture prompt, completion, and total counts for this model call."""
        response = SimpleNamespace(
            content="The policy grants 12 days.",
            usage_metadata={
                "input_tokens": 120,
                "output_tokens": 9,
                "total_tokens": 129,
            },
        )
        response_model = SimpleNamespace(invoke=lambda _messages: response)

        with patch("pdf_rag.chat_model._active_provider", "claude"), patch(
            "pdf_rag.chat_model._active_model", "claude-sonnet-4-5"
        ), patch("pdf_rag.chat_model._create_chat_model", return_value=response_model):
            with collect_model_usage() as usage_records:
                invoke_chat(
                    [{"role": "user", "content": "How many days?"}],
                    operation="answer_draft",
                )

        self.assertEqual(len(usage_records), 1)
        self.assertEqual(usage_records[0]["operation"], "answer_draft")
        self.assertEqual(usage_records[0]["input_tokens"], 120)
        self.assertEqual(usage_records[0]["output_tokens"], 9)
        self.assertEqual(usage_records[0]["total_tokens"], 129)

    def test_invoke_marks_provider_usage_unavailable(self):
        """Report missing provider token metadata as unavailable, not zero."""
        response_model = SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(content="An answer.")
        )

        with patch("pdf_rag.chat_model._active_provider", "claude"), patch(
            "pdf_rag.chat_model._active_model", "claude-sonnet-4-5"
        ), patch("pdf_rag.chat_model._create_chat_model", return_value=response_model):
            with collect_model_usage() as usage_records:
                invoke_chat([{"role": "user", "content": "Question?"}])

        self.assertFalse(usage_records[0]["available"])
        self.assertNotIn("total_tokens", usage_records[0])

    def test_claude_stream_failure_before_first_chunk_falls_back(self):
        """Start the answer from Ollama if Claude fails before streaming text."""
        def failing_stream(_messages):
            raise ConnectionError("offline")
            yield

        claude_model = SimpleNamespace(stream=failing_stream)
        ollama_model = SimpleNamespace(
            stream=lambda _messages: iter([SimpleNamespace(content="fallback stream")])
        )

        with patch("pdf_rag.chat_model._active_provider", "claude"), patch(
            "pdf_rag.chat_model._active_model", "claude-sonnet-4-5"
        ), patch(
            "pdf_rag.chat_model._create_chat_model",
            side_effect=[claude_model, ollama_model],
        ):
            from pdf_rag.chat_model import stream_chat

            chunks = list(stream_chat([{"role": "user", "content": "Question"}]))

        self.assertEqual(chunks, ["fallback stream"])


if __name__ == "__main__":
    unittest.main()