import asyncio

from aperag.docparser.health import get_parser_health_report


def test_parser_health_report_marks_object_store_dependency(monkeypatch):
    monkeypatch.setattr("aperag.docparser.health._package_status", lambda _name: ("ok", "Installed (test)"))
    monkeypatch.setattr("aperag.docparser.health.get_soffice_cmd", lambda: "/usr/bin/soffice")
    monkeypatch.setattr(
        "aperag.docparser.health.get_object_store",
        lambda: (_ for _ in ()).throw(RuntimeError("s3 init failed")),
    )

    report = asyncio.run(get_parser_health_report({"use_markitdown": True, "use_mineru": False}))

    object_store_item = next(item for item in report.dependencies if item.key == "object_store")
    assert object_store_item.status == "error"
    assert "s3 init failed" in object_store_item.detail
    assert any("object store" in warning.lower() for warning in report.warnings)


def test_parser_health_report_explicitly_marks_mineru_as_enhancement(monkeypatch):
    monkeypatch.setattr("aperag.docparser.health._package_status", lambda _name: ("ok", "Installed (test)"))
    monkeypatch.setattr("aperag.docparser.health.get_soffice_cmd", lambda: "/usr/bin/soffice")
    monkeypatch.setattr("aperag.docparser.health.get_object_store", lambda: object())
    monkeypatch.setattr(
        "aperag.docparser.health._probe_mineru", lambda _token: asyncio.sleep(0, result=("ok", "Reachable"))
    )
    monkeypatch.setattr(
        "aperag.docparser.health._probe_paddleocr",
        lambda _host: asyncio.sleep(0, result=("disabled", "Not configured.")),
    )
    monkeypatch.setattr(
        "aperag.docparser.health._probe_whisper",
        lambda _host: asyncio.sleep(0, result=("disabled", "Not configured.")),
    )

    report = asyncio.run(
        get_parser_health_report(
            {
                "use_markitdown": True,
                "use_mineru": True,
                "mineru_api_token": "token",
            }
        )
    )

    mineru_tier = next(tier for tier in report.support_tiers if tier.key == "mineru_enhancement")
    default_tier = next(tier for tier in report.support_tiers if tier.key == "default_local")

    assert default_tier.status == "available"
    assert ".csv" in default_tier.formats
    assert ".xml" in default_tier.formats
    assert ".eml" in default_tier.formats
    assert mineru_tier.status == "available"
    assert "fallback" in mineru_tier.detail.lower()
    assert "not the primary parser path" in mineru_tier.detail.lower()
    assert any("enhancement path" in warning.lower() for warning in report.warnings)
    assert any("enhancement fallback" in recommendation.lower() for recommendation in report.recommendations)
