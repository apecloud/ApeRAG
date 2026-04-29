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


# ---------------------------------------------------------------------
# 任务 #5 step 10 / huangzhangshu task #3 hard-gate (msg=23a23245):
# 文档整体状态聚合不能让 graph_vectors 参与 — 设计文档 §4.5.
# ---------------------------------------------------------------------


def test_document_status_excludes_graph_vectors_from_aggregation_when_facts_active():
    """事实层 ACTIVE + 向量层 PENDING / RUNNING / FAILED — 文档整体应该
    判 COMPLETE, 向量层不参与聚合.
    """
    for vectors_status in ("PENDING", "RUNNING", "FAILED"):
        assert (
            _index_statuses_to_document_status(
                {
                    "VECTOR": {"status": "ACTIVE", "is_serving": True},
                    "FULLTEXT": {"status": "ACTIVE", "is_serving": True},
                    "GRAPH": None,
                    "GRAPH_FACTS": {"status": "ACTIVE", "is_serving": True},
                    "GRAPH_VECTORS": {"status": vectors_status, "is_serving": False},
                    "SUMMARY": None,
                    "VISION": None,
                },
                fallback=DocumentStatus.PENDING,
            )
            == DocumentStatus.COMPLETE
        ), f"vectors_status={vectors_status!r} 不应该把文档整体卡住"


def test_document_status_legacy_graph_only_compatible_completion():
    """兼容期: 只有老 GRAPH 数据 (没有 GRAPH_FACTS 行), 老 GRAPH ACTIVE
    就算文档整体 COMPLETE — 不需要等到全模态都有.
    """
    assert (
        _index_statuses_to_document_status(
            {
                "VECTOR": {"status": "ACTIVE", "is_serving": True},
                "FULLTEXT": {"status": "ACTIVE", "is_serving": True},
                "GRAPH": {"status": "ACTIVE", "is_serving": True},
                "GRAPH_FACTS": None,
                "GRAPH_VECTORS": None,
                "SUMMARY": None,
                "VISION": None,
            },
            fallback=DocumentStatus.PENDING,
        )
        == DocumentStatus.COMPLETE
    )


def test_document_status_facts_failed_takes_priority_over_vectors_active():
    """事实层 FAILED 必须把文档整体判 FAILED, 即使向量层 ACTIVE."""
    assert (
        _index_statuses_to_document_status(
            {
                "VECTOR": {"status": "ACTIVE", "is_serving": True},
                "GRAPH_FACTS": {"status": "FAILED", "is_serving": False},
                "GRAPH_VECTORS": {"status": "ACTIVE", "is_serving": True},
            },
            fallback=DocumentStatus.PENDING,
        )
        == DocumentStatus.FAILED
    )
