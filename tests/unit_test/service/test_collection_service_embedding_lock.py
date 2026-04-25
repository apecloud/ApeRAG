# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for CollectionService.embedding-lock logic.

We intentionally test the two pure helpers and the method that glues them
together (``_reject_embedding_change``) without touching the database: the
guardrail lives entirely above the ORM and is what users will hit when they
try to repoint an existing collection at a new embedding model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aperag.domains.knowledge_base.schemas import CollectionUpdate
from aperag.domains.knowledge_base.service.collection_service import CollectionService
from aperag.exceptions import ValidationException
from aperag.schema.common import CollectionConfig, ModelSpec


def _cfg(model: str | None) -> CollectionConfig:
    return CollectionConfig(
        embedding=ModelSpec(model_id=model),
    )


def _instance_with(cfg: CollectionConfig | None):
    """Fake SQLAlchemy row; the service only reads ``.config`` as JSON text."""
    return SimpleNamespace(config=None if cfg is None else cfg.model_dump_json())


# ---------------------------------------------------------------------------
# _embedding_identity: the building block for comparison
# ---------------------------------------------------------------------------


def test_embedding_identity_none_for_no_config():
    assert CollectionService._embedding_identity(None) is None


def test_embedding_identity_none_when_embedding_missing():
    assert CollectionService._embedding_identity(CollectionConfig()) is None


def test_embedding_identity_none_when_all_embedding_fields_empty():
    # Empty ModelSpec should not count as "bound" — lets first-time binding happen.
    assert CollectionService._embedding_identity(_cfg(None)) is None


def test_embedding_identity_is_tuple():
    ident = CollectionService._embedding_identity(_cfg("mdl-bge-m3"))
    assert ident == ("mdl-bge-m3",)


# ---------------------------------------------------------------------------
# _reject_embedding_change: the guardrail itself
# ---------------------------------------------------------------------------


def _update(cfg: CollectionConfig | None) -> CollectionUpdate:
    return CollectionUpdate(config=cfg)


@pytest.fixture
def svc() -> CollectionService:
    # Avoid touching the global db_ops singleton — the guardrail is pure.
    s = CollectionService.__new__(CollectionService)
    s.db_ops = None
    return s


def test_reject_allows_first_time_binding(svc):
    """A collection whose embedding was never set can receive one."""
    instance = _instance_with(None)
    svc._reject_embedding_change(instance, _update(_cfg("bge-m3")))  # no raise


def test_reject_allows_identical_embedding(svc):
    """Saving the same embedding block (e.g. title/description edits) is fine."""
    cfg = _cfg("mdl-bge-m3")
    instance = _instance_with(cfg)
    svc._reject_embedding_change(instance, _update(cfg))  # no raise


def test_reject_allows_update_without_config_field(svc):
    """Partial updates (``config=None``) must not trigger the guardrail;
    otherwise every title-only edit would fail validation."""
    instance = _instance_with(_cfg("mdl-bge-m3"))
    svc._reject_embedding_change(instance, CollectionUpdate(config=None))


def test_reject_blocks_model_change(svc):
    """Switching the embedding model itself is the headline offense."""
    instance = _instance_with(_cfg("mdl-bge-m3"))
    with pytest.raises(ValidationException, match="cannot be changed"):
        svc._reject_embedding_change(instance, _update(_cfg("text-embedding-3-large")))


def test_reject_blocks_clearing_existing_binding(svc):
    """Once bound, the embedding block cannot be wiped clean — doing so would
    leave existing Qdrant points stranded with no way to refresh them."""
    instance = _instance_with(_cfg("mdl-bge-m3"))
    empty = CollectionConfig()  # embedding=None
    with pytest.raises(ValidationException, match="cannot be cleared"):
        svc._reject_embedding_change(instance, _update(empty))


def test_reject_handles_malformed_stored_config_as_unbound(svc):
    """Historical rows with garbage in ``config`` should not brick future
    updates; we treat unparseable ``instance.config`` as "no prior binding"
    so the user can set one via the normal edit flow."""
    instance = SimpleNamespace(config="{not valid json")
    # No raise — treated as first-time binding.
    svc._reject_embedding_change(instance, _update(_cfg("mdl-bge-m3")))


def test_reject_accepts_future_schema_additions_identically(svc):
    """The identity tuple is based solely on model_id; unrelated
    ModelSpec fields (temperature etc.) must not trigger the guardrail.
    Otherwise users can't tune retrieval params post-hoc."""
    before = ModelSpec(model_id="mdl-bge-m3", temperature=0.0)
    after = ModelSpec(model_id="mdl-bge-m3", temperature=0.5)
    instance = _instance_with(CollectionConfig(embedding=before))
    svc._reject_embedding_change(instance, _update(CollectionConfig(embedding=after)))  # no raise


def test_reject_parses_stored_config_correctly(svc):
    """Sanity: the guardrail really reads embedding from the serialized JSON
    in ``instance.config`` rather than assuming a richer in-memory shape."""
    stored = json.loads(_cfg("mdl-bge-m3").model_dump_json())
    instance = SimpleNamespace(config=json.dumps(stored))
    with pytest.raises(ValidationException):
        svc._reject_embedding_change(instance, _update(_cfg("text-embedding-3-large")))
