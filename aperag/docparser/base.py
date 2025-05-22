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
    pass


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


class AssetBinPart(BaseModel):
    asset_id: str
    data: bytes
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseParser(ABC):
    @abstractmethod
    def supported_extensions(self) -> list[str]: ...

    def accept(self, extension: str) -> bool:
        return extension.lower() in self.supported_extensions()

    @abstractmethod
    def parse_file(self, path: Path, metadata: dict[str, Any] = {}, **kwargs) -> list[Part]: ...
