"""Tests for ``CollectionTask`` graceful skip paths in ``_initialize_vector_databases``.

Mirrors the existing ``_initialize_fulltext_index`` ``enable_fulltext`` skip
behavior: a collection with ``enable_vector=false`` must not trigger an
embedding provider lookup, otherwise provider-independent collections cause
a Celery retry storm via ``model_service_provider`` ``NoneType`` access.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from aperag.tasks.collection import CollectionTask


def _collection(*, enable_vector: bool, embedding=None):
    config = {
        "source": "system",
        "enable_vector": enable_vector,
        "enable_fulltext": False,
        "enable_knowledge_graph": False,
        "enable_summary": False,
        "enable_vision": False,
    }
    if embedding is not None:
        config["embedding"] = embedding
    return SimpleNamespace(id="coll-1", config=json.dumps(config), user="user-1")


def test_initialize_vector_databases_skips_when_enable_vector_false():
    """enable_vector=false collection skips embedding lookup entirely."""
    task = CollectionTask()
    collection = _collection(enable_vector=False)

    with (
        patch("aperag.tasks.collection.get_collection_embedding_service_sync") as mock_emb,
        patch("aperag.tasks.collection.get_vector_db_connector") as mock_vdb,
    ):
        task._initialize_vector_databases("coll-1", collection)

    mock_emb.assert_not_called()
    mock_vdb.assert_not_called()


def test_initialize_vector_databases_resolves_embedding_when_enable_vector_true():
    """enable_vector=true collection resolves embedding provider as before."""
    task = CollectionTask()
    collection = _collection(
        enable_vector=True,
        embedding={
            "model": "text-embedding-3-small",
            "model_service_provider": "openai",
            "custom_llm_provider": "openai",
        },
    )

    fake_connector = SimpleNamespace(connector=SimpleNamespace(ensure_collection=lambda: None))
    with (
        patch(
            "aperag.tasks.collection.get_collection_embedding_service_sync",
            return_value=(SimpleNamespace(), 1536),
        ) as mock_emb,
        patch(
            "aperag.tasks.collection.get_vector_db_connector",
            return_value=fake_connector,
        ) as mock_vdb,
    ):
        task._initialize_vector_databases("coll-1", collection)

    mock_emb.assert_called_once_with(collection)
    assert mock_vdb.call_count == 1
    _, kwargs = mock_vdb.call_args
    assert kwargs["vector_size"] == 1536
