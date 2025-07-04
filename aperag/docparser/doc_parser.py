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

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from aperag.docparser.audio_parser import AudioParser
from aperag.docparser.base import BaseParser, FallbackError, Part
from aperag.docparser.docray_parser import DocRayParser
from aperag.docparser.image_parser import ImageParser
from aperag.docparser.markitdown_parser import MarkItDownParser
from aperag.docparser.mineru_parser import MinerUParser

logger = logging.getLogger(__name__)

ALL_PARSERS = [
    AudioParser,
    ImageParser,
    MarkItDownParser,
    DocRayParser,
    MinerUParser,
]

PARSER_MAP = {cls.name: cls for cls in ALL_PARSERS}


def get_default_config() -> list["ParserConfig"]:
    return [
        ParserConfig(name=MinerUParser.name, enabled=False),
        ParserConfig(name=DocRayParser.name, enabled=True),
        ParserConfig(name=ImageParser.name, enabled=True),
        ParserConfig(name=AudioParser.name, enabled=True),
        ParserConfig(name=MarkItDownParser.name, enabled=True),
    ]


def get_fast_doc_parser() -> "DocParser":
    fast_config = [
        ParserConfig(name=ImageParser.name, enabled=True),
        ParserConfig(name=AudioParser.name, enabled=True),
        ParserConfig(name=MarkItDownParser.name, enabled=True),
    ]
    return DocParser(config=fast_config)


class ParserConfig(BaseModel):
    name: str = Field(..., description="The name of the parser")
    enabled: bool = Field(True, description="Whether this parser is enabled")
    supported_extensions_override: Optional[List[str]] = Field(
        None, description="Override the supported file extensions for the parser. If None, no override."
    )
    settings: Optional[dict[str, Any]] = Field(None, description="Other settings for the parser, e.g., API key.")


class DocParser(BaseParser):
    def __init__(self, parser_config: Optional[dict] = None, full_config: list[ParserConfig] = None):
        self.config = full_config or get_default_config()
        self.supported = None
        self.parsing_order: list[str] = []
        self.parsers: dict[str, BaseParser] = {}
        self.ext_override = {}

        parser_config = parser_config or {}

        # Dynamically update parser configs based on collection settings
        for cfg in self.config:
            if cfg.name == MinerUParser.name:
                use_mineru = parser_config.get("use_mineru", False)
                if use_mineru:
                    cfg.enabled = True

                token_from_env = os.getenv("MINERU_API_TOKEN")
                token = parser_config.get("mineru_api_token") or token_from_env
                if token is not None:
                    if cfg.settings is None:
                        cfg.settings = {}
                    cfg.settings["api_token"] = token

        for cfg in self.config:
            if not cfg.enabled:
                continue
            parser_class = PARSER_MAP.get(cfg.name)
            if parser_class is None:
                # Parser not found
                logger.warning(f'Parser "{cfg.name}" not found. Skipping.')
                continue
            if cfg.supported_extensions_override is not None:
                self.ext_override[cfg.name] = cfg.supported_extensions_override
            parser = parser_class(**(cfg.settings or {}))
            self.parsing_order.append(cfg.name)
            self.parsers[cfg.name] = parser

    def _get_parser_supported_extensions(self, parser_name: str) -> list[str]:
        if self.parsers.get(parser_name) is None:
            return []
        base = self.parsers[parser_name].supported_extensions()
        if self.ext_override.get(parser_name) is None:
            return base
        return [ext for ext in base if ext in self.ext_override[parser_name]]

    def _parser_accept(self, parser_name: str, extension: str) -> bool:
        supported = self._get_parser_supported_extensions(parser_name)
        return extension in supported

    def supported_extensions(self) -> list[str]:
        if self.supported is not None:
            return self.supported
        supported = []
        for parser_name in self.parsers.keys():
            supported.extend(self._get_parser_supported_extensions(parser_name))
        self.supported = sorted(list(set(supported)))
        return self.supported

    def parse_file(self, path: Path, metadata: dict[str, Any] = {}, **kwargs) -> list[Part]:
        extension = path.suffix
        last_err = None
        for parser_name in self.parsing_order:
            parser = self.parsers[parser_name]
            if not self._parser_accept(parser_name, extension):
                continue
            try:
                return parser.parse_file(path, metadata, **kwargs)
            except FallbackError as e:
                last_err = e
        raise ValueError(f'No parser can handle file with extension "{extension}"') from last_err
