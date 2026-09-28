"""Tests for registry-backed document version comparisons."""

import unittest
from unittest.mock import patch

from pdf_rag.chat import (
    build_version_comparison_context,
    find_document_name_in_query,
    get_requested_version_pairs,
    is_version_comparison_query,
)
from pdf_rag.document_registry import get_document_version_changes


class DocumentComparisonTests(unittest.TestCase):
    """Verify document and version selection for change comparison requests."""

    def test_detects_comparison_requests(self):
        """Recognize common natural language requests for revision changes."""
        self.assertTrue(
            is_version_comparison_query(
                "difference between old and latest version of Hackathon document"
            )
        )
        self.assertFalse(is_version_comparison_query("Who wrote the Hackathon document?"))

    def test_matches_document_names_with_spaces(self):
        """Resolve the full registry filename rather than truncating at a space."""
        document_name = find_document_name_in_query(
            "difference between versions for hackathon document",
            ["Hackathon 2026.pdf", "Leave_Policy.pdf"],
        )
        self.assertEqual(document_name, "Hackathon 2026.pdf")

    def test_defaults_to_every_adjacent_version_transition(self):
        """Compare each successive revision when the user gives no explicit pair."""
        self.assertEqual(
            get_requested_version_pairs("show changes at each version", [1, 2, 3, 4]),
            [(1, 2), (2, 3), (3, 4)],
        )

    def test_selects_an_explicit_version_pair(self):
        """Honor a concrete request for two numbered versions."""
        self.assertEqual(
            get_requested_version_pairs("compare v1 versus version 3", [1, 2, 3]),
            [(1, 3)],
        )

    def test_builds_only_added_and_removed_text_per_transition(self):
        """Build separated transition context from registry changes, not search results."""
        registry = {
            "Hackathon 2026.pdf": {
                "history": {"1": {}, "2": {}, "3": {}},
            }
        }
        first_change = {
            "change_sections": [{"removed": "Old date: May", "added": "New date: June"}]
        }
        second_change = {
            "change_sections": [{"removed": "Old venue: Room A", "added": "New venue: Room B"}]
        }
        with patch("pdf_rag.chat.load_registry", return_value=registry), patch(
            "pdf_rag.chat.get_document_version_changes",
            side_effect=[first_change, second_change],
        ):
            context, citations = build_version_comparison_context(
                "show changes at each version for hackathon"
            )

        self.assertIn("Removed from v1:\nOld date: May", context)
        self.assertIn("Added in v2:\nNew date: June", context)
        self.assertIn("Removed from v2:\nOld venue: Room A", context)
        self.assertIn("Added in v3:\nNew venue: Room B", context)
        self.assertEqual(len(citations), 2)

    def test_registry_change_sections_omit_unchanged_lines(self):
        """Extract only the changed lines from two saved registry versions."""
        registry = {
            "policy.pdf": {
                "history": {
                    "1": {"full_text": "Same\nOld rule\nUnchanged", "timestamp": "t1"},
                    "2": {"full_text": "Same\nNew rule\nUnchanged", "timestamp": "t2"},
                }
            }
        }
        with patch("pdf_rag.document_registry.load_registry", return_value=registry):
            changes = get_document_version_changes("policy.pdf", 1, 2)

        self.assertEqual(
            changes["change_sections"],
            [{"removed": "Old rule", "added": "New rule"}],
        )


if __name__ == "__main__":
    unittest.main()