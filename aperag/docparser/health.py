# Copyright 2025 ApeCloud, Inc.
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

import asyncio
import base64
import io
import wave
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from aperag.config import settings
from aperag.docparser.audio_parser import SUPPORTED_EXTENSIONS as AUDIO_EXTENSIONS
from aperag.docparser.doc_parser import DocParser
from aperag.docparser.image_parser import SUPPORTED_EXTENSIONS as IMAGE_EXTENSIONS
from aperag.docparser.markitdown_parser import SUPPORTED_EXTENSIONS as MARKITDOWN_EXTENSIONS
from aperag.docparser.mineru_parser import API_HOST as MINERU_API_HOST
from aperag.docparser.mineru_parser import SUPPORTED_EXTENSIONS as MINERU_EXTENSIONS
from aperag.docparser.utils import get_soffice_cmd

OFFICIAL_FORMATS = [ext for ext in MARKITDOWN_EXTENSIONS if ext not in [".doc", ".ppt"]]
LEGACY_OFFICE_FORMATS = [".doc", ".ppt"]


class ParserHealthItem(BaseModel):
    key: str
    label: str
    status: Literal["ok", "warning", "error", "disabled"]
    detail: str


class ParserSupportTier(BaseModel):
    key: str
    label: str
    category: Literal["official", "conditional", "enhanced", "optional"]
    parser: str
    formats: list[str] = Field(default_factory=list)
    status: Literal["available", "limited", "unavailable", "disabled"]
    detail: str
    requirements: list[str] = Field(default_factory=list)


class ParserHealthReport(BaseModel):
    default_parser: str
    parser_order: list[str] = Field(default_factory=list)
    available_extensions: list[str] = Field(default_factory=list)
    dependencies: list[ParserHealthItem] = Field(default_factory=list)
    services: list[ParserHealthItem] = Field(default_factory=list)
    support_tiers: list[ParserSupportTier] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


def _normalize_parser_settings(parser_settings: dict[str, Any] | None) -> dict[str, Any]:
    parser_settings = parser_settings or {}
    return {
        "use_mineru": parser_settings.get("use_mineru", False),
        "mineru_api_token": parser_settings.get("mineru_api_token") or None,
        "use_markitdown": parser_settings.get("use_markitdown", True),
    }


def _service_url(base_url: str, suffix: str = "") -> str:
    return base_url.rstrip("/") + suffix


def _sample_png_base64() -> str:
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wn7P9sAAAAASUVORK5CYII="
    )
    return base64.b64encode(png_bytes).decode("utf-8")


def _sample_wav_file() -> tuple[str, bytes, str]:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)
    return ("health-check.wav", buffer.getvalue(), "audio/wav")


async def _probe_http_endpoint(
    url: str,
    *,
    method: str = "GET",
    ok_statuses: set[int] | None = None,
    json_body: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[str, str]:
    ok_statuses = ok_statuses or {200}
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            response = await client.request(
                method,
                url,
                json=json_body,
                files=files,
                params=params,
            )
        if response.status_code in ok_statuses:
            return "ok", f"Reachable (HTTP {response.status_code})"
        if 400 <= response.status_code < 500:
            return "error", f"Endpoint responded with HTTP {response.status_code}"
        return "warning", f"Endpoint responded with HTTP {response.status_code}"
    except Exception as e:
        return "warning", str(e)


async def _probe_mineru(token: str | None) -> tuple[str, str]:
    if not token:
        return "disabled", "MinerU token is not configured."

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{MINERU_API_HOST}/api/v4/extract-results/batch/test-token",
                headers={"Authorization": f"Bearer {token}"},
            )
        if response.status_code == 401:
            return "error", "Configured token is invalid."
        if 200 <= response.status_code < 300:
            return "ok", f"Reachable (HTTP {response.status_code})"
        if 400 <= response.status_code < 500:
            return "error", f"MinerU responded with HTTP {response.status_code}"
        return "warning", f"MinerU responded with HTTP {response.status_code}"
    except Exception as e:
        return "warning", f"Token configured but MinerU is unreachable: {e}"


async def _probe_paddleocr(base_url: str) -> tuple[str, str]:
    return await _probe_http_endpoint(
        _service_url(base_url, "/predict/ocr_system"),
        method="POST",
        ok_statuses={200},
        json_body={"images": [_sample_png_base64()]},
    )


async def _probe_whisper(base_url: str) -> tuple[str, str]:
    return await _probe_http_endpoint(
        _service_url(base_url, "/asr"),
        method="POST",
        ok_statuses={200},
        params={
            "encode": "true",
            "task": "transcribe",
            "vad_filter": "true",
            "word_timestamps": "true",
            "output": "txt",
        },
        files={"audio_file": _sample_wav_file()},
    )


def _package_status(package_name: str) -> tuple[str, str]:
    try:
        return "ok", f"Installed ({version(package_name)})"
    except PackageNotFoundError:
        return "error", "Not installed"


async def get_parser_health_report(parser_settings: dict[str, Any] | None = None) -> ParserHealthReport:
    parser_settings = _normalize_parser_settings(parser_settings)
    parser = DocParser(parser_config=parser_settings)
    parser_order = parser.parsing_order
    available_extensions = parser.supported_extensions()

    markitdown_enabled = parser_settings["use_markitdown"]
    mineru_enabled = parser_settings["use_mineru"]
    mineru_token = parser_settings["mineru_api_token"]
    soffice_cmd = get_soffice_cmd()

    markitdown_status, markitdown_detail = _package_status("markitdown")
    dependencies = [
        ParserHealthItem(
            key="markitdown",
            label="MarkItDown package",
            status=markitdown_status,
            detail=markitdown_detail,
        ),
        ParserHealthItem(
            key="soffice",
            label="LibreOffice soffice",
            status="ok" if soffice_cmd else "warning",
            detail=soffice_cmd or "Not found. Legacy .doc/.ppt parsing will be unavailable.",
        ),
    ]

    paddle_host = settings.paddleocr_host
    whisper_host = settings.whisper_host
    mineru_result, paddle_result, whisper_result = await asyncio.gather(
        _probe_mineru(mineru_token) if mineru_enabled else asyncio.sleep(0, result=("disabled", "MinerU is disabled.")),
        _probe_paddleocr(paddle_host)
        if paddle_host
        else asyncio.sleep(0, result=("disabled", "Not configured. Image OCR is disabled.")),
        _probe_whisper(whisper_host)
        if whisper_host
        else asyncio.sleep(0, result=("disabled", "Not configured. Audio transcription is disabled.")),
    )

    mineru_status, mineru_detail = mineru_result
    paddle_status, paddle_detail = paddle_result
    whisper_status, whisper_detail = whisper_result

    services = [
        ParserHealthItem(
            key="mineru",
            label="MinerU enhancement service",
            status=mineru_status,
            detail=mineru_detail,
        ),
        ParserHealthItem(
            key="paddleocr",
            label="PaddleOCR service",
            status=paddle_status,
            detail=paddle_detail,
        ),
        ParserHealthItem(
            key="whisper",
            label="Whisper ASR service",
            status=whisper_status,
            detail=whisper_detail,
        ),
    ]

    support_tiers = [
        ParserSupportTier(
            key="default_local",
            label="Default local parsing",
            category="official",
            parser="markitdown",
            formats=OFFICIAL_FORMATS,
            status="available" if markitdown_enabled and markitdown_status == "ok" else "unavailable",
            detail="Default private-deployment parser path. Uses local MarkItDown without cloud dependency.",
            requirements=["markitdown"],
        ),
        ParserSupportTier(
            key="legacy_office",
            label="Legacy Office conversion",
            category="conditional",
            parser="markitdown+soffice",
            formats=LEGACY_OFFICE_FORMATS,
            status="available" if markitdown_enabled and bool(soffice_cmd) else "limited",
            detail="Legacy .doc/.ppt support requires the soffice binary to convert files before parsing.",
            requirements=["markitdown", "soffice"],
        ),
        ParserSupportTier(
            key="mineru_enhancement",
            label="Complex document enhancement",
            category="enhanced",
            parser="mineru",
            formats=MINERU_EXTENSIONS,
            status=(
                "available" if mineru_enabled and mineru_status == "ok" else "limited" if mineru_enabled else "disabled"
            ),
            detail="Optional enhancement path for complex PDFs and layout-heavy documents. Disabled by default.",
            requirements=["use_mineru=true", "mineru_api_token"],
        ),
        ParserSupportTier(
            key="image_ocr",
            label="Image OCR",
            category="optional",
            parser="image",
            formats=IMAGE_EXTENSIONS,
            status="available" if paddle_status == "ok" else "disabled" if not paddle_host else "limited",
            detail="Optional OCR path for standalone images.",
            requirements=["PADDLEOCR_HOST"],
        ),
        ParserSupportTier(
            key="audio_asr",
            label="Audio transcription",
            category="optional",
            parser="audio",
            formats=AUDIO_EXTENSIONS,
            status="available" if whisper_status == "ok" else "disabled" if not whisper_host else "limited",
            detail="Optional speech-to-text path for uploaded audio files.",
            requirements=["WHISPER_HOST"],
        ),
    ]

    warnings: list[str] = []
    recommendations: list[str] = []

    if not markitdown_enabled:
        warnings.append("MarkItDown is disabled. The default parser path is unavailable for most document formats.")
        recommendations.append("Keep MarkItDown enabled for the default private-deployment parser path.")

    if not soffice_cmd:
        warnings.append("Legacy Office files (.doc/.ppt) are not available because soffice is missing.")
        recommendations.append("Install LibreOffice/OpenOffice if customers need legacy Office support.")

    if mineru_enabled:
        warnings.append(
            "MinerU is enabled. Parsing results may differ from the default local parser for supported files."
        )
        if mineru_status != "ok":
            warnings.append(f"MinerU enhancement is enabled but not healthy: {mineru_detail}")
        recommendations.append("Keep MinerU as an explicit enhancement path, not the default delivery path.")

    if paddle_host and paddle_status != "ok":
        warnings.append("PaddleOCR is configured but currently unreachable or unhealthy.")
    if whisper_host and whisper_status != "ok":
        warnings.append("Whisper ASR is configured but currently unreachable or unhealthy.")

    recommendations.append("Use the support tiers below as the customer-facing parser support matrix.")
    recommendations.append("Prefer deployment-time parser preflight over upload-time trial and error.")

    return ParserHealthReport(
        default_parser="markitdown" if markitdown_enabled else "none",
        parser_order=parser_order,
        available_extensions=available_extensions,
        dependencies=dependencies,
        services=services,
        support_tiers=support_tiers,
        warnings=warnings,
        recommendations=recommendations,
    )
