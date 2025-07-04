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


class FallbackError(Exception):
    pass


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
