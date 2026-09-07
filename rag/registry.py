# rag/registry.py

import json
from pathlib import Path


REGISTRY_PATH = Path("vector_store/document_registry.json")


def load_registry():
    """
    Load information about already indexed documents.
    """

    if not REGISTRY_PATH.exists():
        return {}

    with open(REGISTRY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_registry(registry):
    """
    Save indexed document information.
    """

    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REGISTRY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )