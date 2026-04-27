# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cheap deterministic candidate generation for graph curation."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

from aperag.graph_curation.dto import CurationEntity as Entity

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z]+")


@dataclass(frozen=True)
class CandidatePair:
    left_id: str
    right_id: str
    score: float
    signals: dict[str, Any]


def entity_snapshot(entity: Entity) -> dict[str, Any]:
    return {
        "entity_id": entity.entity_id,
        "entity_name": entity.name,
        "entity_type": entity.type,
        "description": entity.description,
        "source_chunk_count": len(entity.source_chunk_ids or ()),
    }


def build_candidate_pairs(
    entities: Sequence[Entity],
    vector_neighbors: dict[str, list[tuple[str, float]]] | None = None,
    *,
    max_pairs_per_entity: int,
    max_total_pairs: int,
) -> list[CandidatePair]:
    if not entities:
        return []

    vector_neighbors = vector_neighbors or {}
    entities_by_id = {entity.entity_id: entity for entity in entities}
    same_type_groups: dict[str, list[Entity]] = defaultdict(list)
    for entity in entities:
        same_type_groups[(entity.type or "").strip().lower()].append(entity)

    pair_signals: dict[tuple[str, str], dict[str, Any]] = {}

    def _upsert_signal(entity_a: Entity, entity_b: Entity, signal_updates: dict[str, Any]) -> None:
        key = _pair_key(entity_a.entity_id, entity_b.entity_id)
        current = pair_signals.setdefault(key, {})
        for name, value in signal_updates.items():
            if isinstance(value, (int, float)) and isinstance(current.get(name), (int, float)):
                current[name] = max(float(current[name]), float(value))
            else:
                current[name] = value

    for group in same_type_groups.values():
        group_size = len(group)
        for idx in range(group_size):
            left = group[idx]
            for jdx in range(idx + 1, group_size):
                right = group[jdx]
                lexical = _lexical_signals(left, right)
                if lexical:
                    _upsert_signal(left, right, lexical)

    for entity_id, neighbors in vector_neighbors.items():
        left = entities_by_id.get(entity_id)
        if left is None:
            continue
        for neighbor_id, score in neighbors:
            right = entities_by_id.get(neighbor_id)
            if right is None or right.entity_id == left.entity_id or right.type != left.type:
                continue
            _upsert_signal(
                left,
                right,
                {
                    "vector_score": float(score),
                    "vector_neighbor": True,
                },
            )

    ranked: list[CandidatePair] = []
    for left_id, right_id in sorted(pair_signals.keys()):
        left = entities_by_id[left_id]
        right = entities_by_id[right_id]
        signals = dict(pair_signals[(left_id, right_id)])
        shared_chunks = len(set(left.source_chunk_ids or ()) & set(right.source_chunk_ids or ()))
        if shared_chunks:
            signals["shared_chunk_count"] = shared_chunks

        score = _pair_score(signals)
        if score <= 0:
            continue
        ranked.append(CandidatePair(left_id=left_id, right_id=right_id, score=score, signals=signals))

    ranked.sort(key=lambda item: (-item.score, item.left_id, item.right_id))
    per_entity_counts: dict[str, int] = defaultdict(int)
    selected: list[CandidatePair] = []
    for pair in ranked:
        if per_entity_counts[pair.left_id] >= max_pairs_per_entity:
            continue
        if per_entity_counts[pair.right_id] >= max_pairs_per_entity:
            continue
        selected.append(pair)
        per_entity_counts[pair.left_id] += 1
        per_entity_counts[pair.right_id] += 1
        if len(selected) >= max_total_pairs:
            break
    return selected


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    return (left_id, right_id) if left_id < right_id else (right_id, left_id)


def _normalize_name(value: str) -> str:
    compact = _NON_ALNUM_RE.sub(" ", value or "").strip().lower()
    return " ".join(compact.split())


def _tokens(value: str) -> set[str]:
    return {token for token in _WORD_RE.findall((value or "").lower()) if token}


def _acronym(value: str) -> str:
    tokens = [token for token in re.split(r"[\s\-_./]+", (value or "").strip()) if token]
    return "".join(token[0].lower() for token in tokens if token)


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _lexical_signals(left: Entity, right: Entity) -> dict[str, Any]:
    left_norm = _normalize_name(left.name)
    right_norm = _normalize_name(right.name)
    if not left_norm or not right_norm:
        return {}

    signals: dict[str, Any] = {}
    if left_norm == right_norm:
        signals["normalized_name_exact"] = True
    elif left_norm in right_norm or right_norm in left_norm:
        signals["normalized_name_contains"] = True

    left_acronym = _acronym(left.name)
    right_acronym = _acronym(right.name)
    if left_acronym and left_acronym == right_acronym and len(left_acronym) >= 2:
        signals["acronym_match"] = True

    name_token_overlap = _jaccard(_tokens(left.name), _tokens(right.name))
    if name_token_overlap >= 0.5:
        signals["name_token_overlap"] = round(name_token_overlap, 3)

    description_overlap = _jaccard(_tokens(left.description), _tokens(right.description))
    if description_overlap >= 0.2:
        signals["description_token_overlap"] = round(description_overlap, 3)

    return signals


def _pair_score(signals: Dict[str, Any]) -> float:
    score = 0.0
    if signals.get("normalized_name_exact"):
        score += 0.65
    if signals.get("normalized_name_contains"):
        score += 0.35
    if signals.get("acronym_match"):
        score += 0.30
    if signals.get("name_token_overlap"):
        score += min(float(signals["name_token_overlap"]) * 0.35, 0.30)
    if signals.get("description_token_overlap"):
        score += min(float(signals["description_token_overlap"]) * 0.20, 0.15)
    if signals.get("shared_chunk_count"):
        score += min(int(signals["shared_chunk_count"]) * 0.20, 0.50)
    if signals.get("vector_score") is not None:
        score += min(max(float(signals["vector_score"]), 0.0) * 0.35, 0.35)
    return round(score, 3)


__all__ = ["CandidatePair", "build_candidate_pairs", "entity_snapshot"]
