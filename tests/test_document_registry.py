"""Tests for the document version registry."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pdf_rag import document_registry


class DocumentRegistryTests(unittest.TestCase):
    """Verify registry persistence, version increments, and generated diffs."""

    def test_register_and_load_document_versions(self):
        """Persist two versions and ensure their text diff is available."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry_path = Path(temporary_directory) / "document_registry.json"
            with patch.object(document_registry, "REGISTRY_PATH", registry_path):
                registry = document_registry.load_registry()
                first_version = document_registry.register_document_version(
                    registry, "policy.pdf", "Original policy text"
                )
                second_version = document_registry.register_document_version(
                    registry, "policy.pdf", "Updated policy text"
                )
                document_registry.save_registry(registry)

                loaded_registry = document_registry.load_registry()
                difference = document_registry.get_document_diff(
                    "policy.pdf", first_version, second_version
                )

        self.assertEqual(loaded_registry["policy.pdf"]["latest_version"], 2)
        self.assertIn("-Original policy text", difference)
        self.assertIn("+Updated policy text", difference)


if __name__ == "__main__":
    unittest.main()