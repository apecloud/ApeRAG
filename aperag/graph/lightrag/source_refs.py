from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .prompt import GRAPH_FIELD_SEP
from .utils import split_string_by_multi_markers


def normalize_source_references(*values: Any) -> list[str]:
    """Return a deterministic list of chunk/source reference IDs.

    The current graph stack persists the same provenance in two shapes:
    - graph storage: `source_id` as `GRAPH_FIELD_SEP`-joined string
    - vector storage: `chunk_ids` as `list[str]`

    This helper gives the codebase one canonical in-memory representation:
    a sorted, unique `list[str]`.
    """

    refs: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            refs.extend(split_string_by_multi_markers(value, [GRAPH_FIELD_SEP]))
            continue
        if isinstance(value, Iterable):
            for item in value:
                if isinstance(item, str):
                    refs.extend(split_string_by_multi_markers(item, [GRAPH_FIELD_SEP]))

    return sorted(dict.fromkeys(ref for ref in refs if ref))


def serialize_source_references(*values: Any) -> str:
    """Serialize source references back to graph-storage string form."""

    return GRAPH_FIELD_SEP.join(normalize_source_references(*values))


def source_references_overlap(source_id: Any, chunk_ids: Any) -> bool:
    """Return True when graph `source_id` and vector/doc chunk refs overlap."""

    if source_id is None or chunk_ids is None:
        return False

    source_refs = set(normalize_source_references(source_id))
    chunk_refs = set(normalize_source_references(chunk_ids))
    return bool(source_refs.intersection(chunk_refs))
