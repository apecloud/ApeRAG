from aperag.domains.knowledge_base.db.models import DocumentStatus
from aperag.domains.knowledge_base.service.document_service import (
    _index_statuses_to_document_status,
    _markdown_path_from_derived_artifact,
)


def test_document_response_status_uses_active_serving_indexes():
    assert (
        _index_statuses_to_document_status(
            {
                "VECTOR": {"status": "ACTIVE", "is_serving": True},
                "FULLTEXT": {"status": "ACTIVE", "is_serving": True},
                "GRAPH": {"status": "ACTIVE", "is_serving": True},
                "SUMMARY": None,
                "VISION": None,
            },
            fallback=DocumentStatus.PENDING,
        )
        == DocumentStatus.COMPLETE
    )


def test_document_response_status_preserves_running_and_failed_indexes():
    assert (
        _index_statuses_to_document_status(
            {
                "VECTOR": {"status": "ACTIVE", "is_serving": True},
                "FULLTEXT": {"status": "RUNNING", "is_serving": False},
            },
            fallback=DocumentStatus.PENDING,
        )
        == DocumentStatus.RUNNING
    )
    assert (
        _index_statuses_to_document_status(
            {
                "VECTOR": {"status": "ACTIVE", "is_serving": True},
                "FULLTEXT": {"status": "FAILED", "is_serving": False},
            },
            fallback=DocumentStatus.PENDING,
        )
        == DocumentStatus.FAILED
    )


def test_markdown_preview_uses_parse_artifact_directory():
    assert (
        _markdown_path_from_derived_artifact("collections/col/documents/doc/derived/parse_abcd/chunks.jsonl")
        == "collections/col/documents/doc/derived/parse_abcd/markdown.md"
    )
