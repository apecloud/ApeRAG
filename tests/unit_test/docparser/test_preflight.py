import asyncio
from pathlib import Path

import pytest

from aperag.docparser.base import ParserChainError, ParserError
from aperag.docparser.doc_parser import DocParser
from aperag.docparser.preflight import run_document_parse_preflight


def test_doc_parser_prefers_markitdown_before_mineru_when_both_enabled():
    parser = DocParser(
        parser_config={
            "use_markitdown": True,
            "use_mineru": True,
            "mineru_api_token": "token",
        }
    )

    assert parser.parsing_order.index("markitdown") < parser.parsing_order.index("mineru")


def test_run_document_parse_preflight_rejects_unsupported_format():
    with pytest.raises(ParserError) as exc_info:
        asyncio.run(run_document_parse_preflight(Path("unsupported.foo"), parser_config={"use_markitdown": True}))

    assert exc_info.value.source == "preflight"
    assert exc_info.value.code == "unsupported_format"


def test_run_document_parse_preflight_blocks_legacy_office_without_soffice(monkeypatch):
    monkeypatch.setattr("aperag.docparser.preflight.get_soffice_cmd", lambda: None)

    with pytest.raises(ParserChainError) as exc_info:
        asyncio.run(
            run_document_parse_preflight(
                Path("legacy.doc"),
                parser_config={"use_markitdown": True, "use_mineru": False},
            )
        )

    assert exc_info.value.source == "preflight"
    assert exc_info.value.code == "parser_chain_failed"
    assert exc_info.value.attempts[0].code == "missing_dependency"


def test_run_document_parse_preflight_allows_legacy_office_with_mineru_fallback(monkeypatch):
    async def _ok_probe(_key, _probe):
        return ("ok", "Reachable")

    monkeypatch.setattr("aperag.docparser.preflight.get_soffice_cmd", lambda: None)
    monkeypatch.setattr("aperag.docparser.preflight._cached_probe", _ok_probe)

    asyncio.run(
        run_document_parse_preflight(
            Path("legacy.doc"),
            parser_config={
                "use_markitdown": True,
                "use_mineru": True,
                "mineru_api_token": "token",
            },
        )
    )


def test_run_document_parse_preflight_reports_object_store_unavailable(monkeypatch):
    monkeypatch.setattr("aperag.docparser.preflight.get_object_store", lambda: (_ for _ in ()).throw(RuntimeError("s3 init failed")))

    with pytest.raises(ParserError) as exc_info:
        asyncio.run(
            run_document_parse_preflight(
                Path("sample.md"),
                parser_config={"use_markitdown": True},
                object_store_base_path="documents/doc-1",
            )
        )

    assert exc_info.value.source == "preflight"
    assert exc_info.value.code == "object_store_unavailable"
