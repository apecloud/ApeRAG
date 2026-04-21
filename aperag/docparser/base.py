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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ParserError(Exception):
    def __init__(
        self,
        message: str,
        *,
        parser_name: str | None = None,
        code: str = "parser_error",
        detail: str | None = None,
        source: str = "runtime",
    ):
        super().__init__(message)
        self.message = message
        self.parser_name = parser_name
        self.code = code
        self.detail = detail
        self.source = source

    def diagnostic_message(self) -> str:
        parts = []
        parts.append(f"source={self.source}")
        if self.parser_name:
            parts.append(f"parser={self.parser_name}")
        parts.append(f"code={self.code}")
        parts.append(self.message)
        if self.detail:
            parts.append(f"detail={self.detail}")
        return ", ".join(parts)


class FallbackError(ParserError):
    def __init__(
        self,
        message: str,
        *,
        parser_name: str | None = None,
        code: str = "fallback",
        detail: str | None = None,
        source: str = "runtime",
    ):
        super().__init__(message, parser_name=parser_name, code=code, detail=detail, source=source)


class ParserAttempt(BaseModel):
    parser_name: str
    status: str
    message: str
    code: str | None = None
    detail: str | None = None


class ParserChainError(ParserError):
    def __init__(
        self,
        extension: str,
        attempts: list[ParserAttempt],
        detail: str | None = None,
        *,
        source: str = "runtime",
    ):
        self.extension = extension
        self.attempts = attempts
        formatted_attempts = "; ".join(
            f"{attempt.parser_name}[{attempt.status}] {attempt.message}" for attempt in attempts
        )
        message = f'No parser produced a usable result for extension "{extension}"'
        if formatted_attempts:
            message += f": {formatted_attempts}"
        super().__init__(message, parser_name=None, code="parser_chain_failed", detail=detail, source=source)


class Part(BaseModel):
    content: str | None = Field(
        default=None,
        description="The parsed content. If None, it means that information extraction has not been performed on this node yet.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class MarkdownPart(Part):
    markdown: str


class PdfPart(Part):
    data: bytes


class TextPart(Part):
    pass


class TitlePart(TextPart):
    level: int


class CodePart(Part):
    lang: str | None = None


class MediaPart(Part):
    url: str
    mime_type: str | None = None


class ImagePart(MediaPart):
    alt_text: str | None = None
    title: str | None = None


class AssetBinPart(Part):
    asset_id: str
    data: bytes
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseParser(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def supported_extensions(self) -> list[str]: ...

    def accept(self, extension: str) -> bool:
        return extension.lower() in self.supported_extensions()

    @abstractmethod
    def parse_file(self, path: Path, metadata: dict[str, Any] = {}, **kwargs) -> list[Part]: ...
