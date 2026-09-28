import os
import json
import difflib
from datetime import datetime

from .config import REGISTRY_PATH

def load_registry():
    """Load document version metadata, returning an empty registry when absent."""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as registry_file:
            return json.load(registry_file)
    return {}

def save_registry(registry):
    """Persist document version metadata as formatted JSON."""
    with open(REGISTRY_PATH, "w", encoding="utf-8") as registry_file:
        json.dump(registry, registry_file, indent=4)

def register_document_version(registry, source_name, full_text):
    """Add an extracted document version and return its assigned version number."""
    if source_name in registry:
        current_version = registry[source_name]["latest_version"] + 1
    else:
        current_version = 1
        registry[source_name] = {"latest_version": 1, "history": {}}

    registry[source_name]["latest_version"] = current_version
    registry[source_name]["history"][str(current_version)] = {
        "timestamp": datetime.now().isoformat(),
        "full_text": full_text
    }
    return current_version

def get_document_diff(source_name, first_version, second_version):
    """Return a unified text diff between two saved versions of a document."""
    registry = load_registry()
    if source_name not in registry:
        return f"File '{source_name}' not found."
    
    history = registry[source_name]["history"]
    if str(first_version) not in history or str(second_version) not in history:
        return f"Specified versions (v{first_version}, v{second_version}) do not exist for '{source_name}'."

    first_text = history[str(first_version)]["full_text"].splitlines()
    second_text = history[str(second_version)]["full_text"].splitlines()

    diff = difflib.unified_diff(
        first_text, second_text,
        fromfile=f"{source_name} (v{first_version})",
        tofile=f"{source_name} (v{second_version})",
        lineterm=""
    )
    return "\n".join(diff)


def get_document_version_changes(source_name, first_version, second_version):
    """Return changed text and version metadata for one document version pair."""
    registry = load_registry()
    if source_name not in registry:
        return None

    history = registry[source_name]["history"]
    first_key = str(first_version)
    second_key = str(second_version)
    if first_key not in history or second_key not in history:
        return None

    first_lines = history[first_key]["full_text"].splitlines()
    second_lines = history[second_key]["full_text"].splitlines()
    matcher = difflib.SequenceMatcher(a=first_lines, b=second_lines, autojunk=False)
    change_sections = []

    for operation, first_start, first_end, second_start, second_end in matcher.get_opcodes():
        if operation == "equal":
            continue
        removed_text = "\n".join(first_lines[first_start:first_end]).strip()
        added_text = "\n".join(second_lines[second_start:second_end]).strip()
        if removed_text or added_text:
            change_sections.append({"removed": removed_text, "added": added_text})

    return {
        "source_name": source_name,
        "from_version": int(first_version),
        "to_version": int(second_version),
        "from_timestamp": history[first_key].get("timestamp"),
        "to_timestamp": history[second_key].get("timestamp"),
        "change_sections": change_sections,
    }