import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from aperag.domains.knowledge_base.schemas import UploadDocumentResponse
from aperag.domains.knowledge_base.service.document_service import DocumentService
from aperag.domains.web_access import schemas as web_access_schemas


class _FakeReaderService:
    plans = []
    calls = []

    def __init__(self, provider_name=None, provider_config=None):
        self.provider_name = provider_name
        self.provider_config = provider_config or {}

    async def __aenter__(self):
        type(self).calls.append((self.provider_name, self.provider_config))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def read(self, request):
        assert request.url_list
        return type(self).plans.pop(0)


def _web_read_response(*results: web_access_schemas.WebReadResultItem) -> web_access_schemas.WebReadResponse:
    succeeded = sum(1 for result in results if result.status == "success")
    return web_access_schemas.WebReadResponse(
        results=list(results),
        total_urls=len(results),
        successful=succeeded,
        failed=len(results) - succeeded,
        processing_time=0.1,
    )


def test_fetch_url_documents_imports_markdown_via_trafilatura_when_no_jina_key(monkeypatch):
    service = DocumentService()
    service.upload_document = AsyncMock(
        return_value=UploadDocumentResponse(
            document_id="doc-fetch-1",
            filename="ACME Report.md",
            size=18,
            status="UPLOADED",
        )
    )

    _FakeReaderService.calls = []
    _FakeReaderService.plans = [
        _web_read_response(
            web_access_schemas.WebReadResultItem(
                url="https://example.com/report",
                status="success",
                title="ACME Report",
                content="# ACME Report\n\nquarterly parser import",
            )
        )
    ]

    monkeypatch.setattr(
        "aperag.domains.model_platform.service.model_service.model_platform_service.get_user_provider_api_key",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr("aperag.domains.web_access.reader.reader_service.ReaderService", _FakeReaderService)

    response = asyncio.run(service.fetch_url_documents("user-1", "col-1", ["https://example.com/report"]))

    assert response.succeeded == 1
    assert response.failed == 0
    assert response.results[0].fetch_status == "success"
    assert response.results[0].document_id == "doc-fetch-1"
    assert _FakeReaderService.calls == [("trafilatura", {})]

    upload_args = service.upload_document.await_args.args
    assert upload_args[:2] == ("user-1", "col-1")
    upload_file = upload_args[2]
    assert upload_file.filename == "ACME Report.md"
    assert upload_file.content_type == "text/markdown"
    assert asyncio.run(upload_file.read()).decode("utf-8") == "# ACME Report\n\nquarterly parser import"


def test_fetch_url_documents_falls_back_when_jina_returns_no_successes(monkeypatch):
    service = DocumentService()
    service.upload_document = AsyncMock(
        return_value=UploadDocumentResponse(
            document_id="doc-fetch-2",
            filename="Fallback Title.md",
            size=12,
            status="UPLOADED",
        )
    )

    _FakeReaderService.calls = []
    _FakeReaderService.plans = [
        _web_read_response(
            web_access_schemas.WebReadResultItem(
                url="https://example.com/fallback",
                status="error",
                error="provider timeout",
                error_code="timeout",
            )
        ),
        _web_read_response(
            web_access_schemas.WebReadResultItem(
                url="https://example.com/fallback",
                status="success",
                title="Fallback Title",
                content="fallback markdown body",
            )
        ),
    ]

    monkeypatch.setattr(
        "aperag.domains.model_platform.service.model_service.model_platform_service.get_user_provider_api_key",
        AsyncMock(return_value="jina-key"),
    )
    monkeypatch.setattr("aperag.domains.web_access.reader.reader_service.ReaderService", _FakeReaderService)

    response = asyncio.run(service.fetch_url_documents("user-1", "col-1", ["https://example.com/fallback"]))

    assert response.succeeded == 1
    assert response.failed == 0
    assert [call[0] for call in _FakeReaderService.calls] == ["jina", "trafilatura"]


def test_fetch_url_documents_falls_back_per_failed_url_when_jina_partially_succeeds(monkeypatch):
    service = DocumentService()
    service.upload_document = AsyncMock(
        side_effect=[
            UploadDocumentResponse(
                document_id="doc-fetch-3a",
                filename="Primary Title.md",
                size=14,
                status="UPLOADED",
            ),
            UploadDocumentResponse(
                document_id="doc-fetch-3b",
                filename="Fallback Title.md",
                size=16,
                status="UPLOADED",
            ),
        ]
    )

    _FakeReaderService.calls = []
    _FakeReaderService.plans = [
        _web_read_response(
            web_access_schemas.WebReadResultItem(
                url="https://example.com/primary",
                status="success",
                title="Primary Title",
                content="primary markdown",
            ),
            web_access_schemas.WebReadResultItem(
                url="https://example.com/fallback",
                status="error",
                error="provider timeout",
                error_code="timeout",
            ),
        ),
        _web_read_response(
            web_access_schemas.WebReadResultItem(
                url="https://example.com/fallback",
                status="success",
                title="Fallback Title",
                content="fallback markdown",
            )
        ),
    ]

    monkeypatch.setattr(
        "aperag.domains.model_platform.service.model_service.model_platform_service.get_user_provider_api_key",
        AsyncMock(return_value="jina-key"),
    )
    monkeypatch.setattr("aperag.domains.web_access.reader.reader_service.ReaderService", _FakeReaderService)

    response = asyncio.run(
        service.fetch_url_documents(
            "user-1",
            "col-1",
            ["https://example.com/primary", "https://example.com/fallback"],
        )
    )

    assert response.succeeded == 2
    assert response.failed == 0
    assert [call[0] for call in _FakeReaderService.calls] == ["jina", "trafilatura"]
    assert [result.url for result in response.results] == [
        "https://example.com/primary",
        "https://example.com/fallback",
    ]
    assert [result.fetch_status for result in response.results] == ["success", "success"]


def test_fetch_url_documents_reports_upload_failures_per_url(monkeypatch):
    service = DocumentService()
    service.upload_document = AsyncMock(side_effect=RuntimeError("quota exceeded"))

    _FakeReaderService.calls = []
    _FakeReaderService.plans = [
        _web_read_response(
            web_access_schemas.WebReadResultItem(
                url="https://example.com/quota",
                status="success",
                title="Quota Title",
                content="quota body",
            )
        )
    ]

    monkeypatch.setattr(
        "aperag.domains.model_platform.service.model_service.model_platform_service.get_user_provider_api_key",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr("aperag.domains.web_access.reader.reader_service.ReaderService", _FakeReaderService)

    response = asyncio.run(service.fetch_url_documents("user-1", "col-1", ["https://example.com/quota"]))

    assert response.succeeded == 0
    assert response.failed == 1
    assert response.results[0].fetch_status == "error"
    assert "quota exceeded" in (response.results[0].error or "")


def test_fetch_url_documents_rejects_more_than_ten_urls():
    service = DocumentService()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(service.fetch_url_documents("user-1", "col-1", [f"https://example.com/{idx}" for idx in range(11)]))

    assert exc_info.value.status_code == 400
    assert "maximum 10 URLs" in exc_info.value.detail
