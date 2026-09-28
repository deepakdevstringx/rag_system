import os
import json
import difflib
from datetime import datetime
from config import REGISTRY_FILE

def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=4)

def update_document_version(registry, filename, full_text):
    """Updates or registers a new version for a given filename."""
    if filename in registry:
        current_version = registry[filename]["latest_version"] + 1
    else:
        current_version = 1
        registry[filename] = {"latest_version": 1, "history": {}}

    registry[filename]["latest_version"] = current_version
    registry[filename]["history"][str(current_version)] = {
        "timestamp": datetime.now().isoformat(),
        "full_text": full_text
    }
    return current_version

def get_document_diff(filename, v1, v2):
    """Calculates diff between version v1 and v2 of a document."""
    registry = load_registry()
    if filename not in registry:
        return f"File '{filename}' not found."
    
    hist = registry[filename]["history"]
    if str(v1) not in hist or str(v2) not in hist:
        return f"Specified versions (v{v1}, v{v2}) do not exist for '{filename}'."

    t1 = hist[str(v1)]["full_text"].splitlines()
    t2 = hist[str(v2)]["full_text"].splitlines()

    diff = difflib.unified_diff(
        t1, t2, 
        fromfile=f"{filename} (v{v1})", 
        tofile=f"{filename} (v{v2})", 
        lineterm=""
    )
    return "\n".join(diff)