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

from pathlib import Path
from typing import Any

from aperag.cache import NAMESPACE_PARSER_PREFLIGHT, application_cache_policy, get_application_cache
from aperag.config import settings
from aperag.docparser.audio_parser import AudioParser
from aperag.docparser.base import ParserAttempt, ParserChainError, ParserError
from aperag.docparser.doc_parser import DocParser
from aperag.docparser.health import _probe_mineru, _probe_paddleocr, _probe_whisper
from aperag.docparser.image_parser import ImageParser
from aperag.docparser.markitdown_parser import MarkItDownParser
from aperag.docparser.mineru_parser import MinerUParser
from aperag.docparser.utils import get_soffice_cmd
from aperag.objectstore.base import get_object_store


def _cache_key(name: str, value: str) -> tuple[str, str]:
    return (name, value)


async def _cached_probe(key: tuple[str, str], probe):
    cache = await get_application_cache()
    result = await cache.get_or_compute(
        namespace=NAMESPACE_PARSER_PREFLIGHT,
        key_data={"probe": key[0], "target": key[1]},
        compute=probe,
        policy=application_cache_policy(NAMESPACE_PARSER_PREFLIGHT),
    )
    return tuple(result)


def _preflight_error(
    message: str, *, code: str, detail: str | None = None, parser_name: str | None = None
) -> ParserError:
    return ParserError(message, code=code, detail=detail, parser_name=parser_name, source="preflight")


async def run_document_parse_preflight(
    path: Path,
    *,
    parser_config: dict[str, Any] | None = None,
    object_store_base_path: str | None = None,
):
    parser = DocParser(parser_config=parser_config)
    extension = path.suffix.lower()

    if not parser.accept(extension):
        raise _preflight_error(
            f'Unsupported file type "{extension or "unknown"}"',
            code="unsupported_format",
            detail="The current parser configuration does not provide a handler for this extension.",
        )

    if object_store_base_path is not None:
        try:
            get_object_store()
        except Exception as e:
            raise _preflight_error(
                "Object store is not available for parsed output persistence",
                code="object_store_unavailable",
                detail=str(e),
            ) from e

    attempts: list[ParserAttempt] = []
    parser_config = parser_config or {}
    relevant_parsers = [name for name in parser.parsing_order if parser._parser_accept(name, extension)]

    for parser_name in relevant_parsers:
        if parser_name == MarkItDownParser.name:
            if extension in {".doc", ".ppt"} and get_soffice_cmd() is None:
                attempts.append(
                    ParserAttempt(
                        parser_name=parser_name,
                        status="blocked",
                        message="Legacy Office conversion requires soffice",
                        code="missing_dependency",
                        detail="Install LibreOffice/OpenOffice or upload .docx/.pptx instead.",
                    )
                )
                continue
            return

        if parser_name == ImageParser.name:
            base_url = settings.paddleocr_host
            if not base_url:
                attempts.append(
                    ParserAttempt(
                        parser_name=parser_name,
                        status="blocked",
                        message="Image OCR is not configured",
                        code="missing_configuration",
                        detail="Set PADDLEOCR_HOST to enable image OCR.",
                    )
                )
                continue
            status, detail = await _cached_probe(_cache_key("paddleocr", base_url), lambda: _probe_paddleocr(base_url))
            if status == "ok":
                return
            attempts.append(
                ParserAttempt(
                    parser_name=parser_name,
                    status="blocked",
                    message="Image OCR preflight failed",
                    code="service_unreachable",
                    detail=detail,
                )
            )
            continue

        if parser_name == AudioParser.name:
            base_url = settings.whisper_host
            if not base_url:
                attempts.append(
                    ParserAttempt(
                        parser_name=parser_name,
                        status="blocked",
                        message="Audio transcription is not configured",
                        code="missing_configuration",
                        detail="Set WHISPER_HOST to enable audio transcription.",
                    )
                )
                continue
            status, detail = await _cached_probe(_cache_key("whisper", base_url), lambda: _probe_whisper(base_url))
            if status == "ok":
                return
            attempts.append(
                ParserAttempt(
                    parser_name=parser_name,
                    status="blocked",
                    message="Audio transcription preflight failed",
                    code="service_unreachable",
                    detail=detail,
                )
            )
            continue

        if parser_name == MinerUParser.name:
            token = parser_config.get("mineru_api_token") or None
            if not token:
                attempts.append(
                    ParserAttempt(
                        parser_name=parser_name,
                        status="blocked",
                        message="MinerU is enabled but no API token is configured",
                        code="missing_configuration",
                        detail="Set mineru_api_token to enable MinerU enhancement.",
                    )
                )
                continue
            status, detail = await _cached_probe(_cache_key("mineru", token), lambda: _probe_mineru(token))
            if status == "ok":
                return
            attempts.append(
                ParserAttempt(
                    parser_name=parser_name,
                    status="blocked",
                    message="MinerU enhancement preflight failed",
                    code="service_unreachable" if status == "warning" else "missing_configuration",
                    detail=detail,
                )
            )
            continue

        return

    if attempts:
        raise ParserChainError(
            extension, attempts, detail="All relevant parsers were blocked during preflight.", source="preflight"
        )
