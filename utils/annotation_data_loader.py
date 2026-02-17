"""Load lightweight annotation JSON files from annotation_data/ directory."""

import json
import os
from typing import Any

import streamlit as st

ANNOTATION_DATA_DIR = "annotation_data"


@st.cache_data
def get_available_annotation_models() -> list[str]:
    """List unique model names from annotation data filenames (excludes example__ files)."""
    if not os.path.exists(ANNOTATION_DATA_DIR):
        return []
    models = set()
    for f in os.listdir(ANNOTATION_DATA_DIR):
        if f.startswith("example__") or not f.endswith(".json") or "__" not in f:
            continue
        model = f.rsplit("__", 1)[0]
        models.add(model)
    return sorted(models)


@st.cache_data
def get_available_annotation_examples(model: str) -> list[str]:
    """List example IDs available for a given model."""
    if not os.path.exists(ANNOTATION_DATA_DIR):
        return []
    examples = []
    prefix = f"{model}__"
    for f in os.listdir(ANNOTATION_DATA_DIR):
        if f.startswith(prefix) and f.endswith(".json"):
            example_id = f[len(prefix):-len(".json")]
            examples.append(example_id)
    try:
        examples.sort(key=lambda x: int(x))
    except ValueError:
        examples.sort()
    return examples


@st.cache_data
def load_annotation_example(model: str, example_id: str) -> dict[str, Any] | None:
    """Load and return the annotation JSON for a model + example."""
    path = os.path.join(ANNOTATION_DATA_DIR, f"{model}__{example_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def get_example_files() -> list[str]:
    """List available tutorial example files (example__*.json)."""
    if not os.path.exists(ANNOTATION_DATA_DIR):
        return []
    examples = []
    for f in sorted(os.listdir(ANNOTATION_DATA_DIR)):
        if f.startswith("example__") and f.endswith(".json"):
            # Strip prefix and extension: example__model__id.json -> model__id
            name = f[len("example__"):-len(".json")]
            examples.append(name)
    return examples


@st.cache_data
def load_example_annotation(name: str) -> dict[str, Any] | None:
    """Load a tutorial example file by its name (model__id)."""
    path = os.path.join(ANNOTATION_DATA_DIR, f"example__{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)
