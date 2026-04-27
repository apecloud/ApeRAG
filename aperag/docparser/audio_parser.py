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

import hashlib
from pathlib import Path
from typing import Any

import requests

from aperag.cache import NAMESPACE_REMOTE_PARSER, application_cache_policy, get_sync_application_cache
from aperag.config import settings
from aperag.docparser.base import BaseParser, FallbackError, ParserError, Part, TextPart

SUPPORTED_EXTENSIONS = [
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
    ".ogg",
    ".flac",
]

REQUEST_TIMEOUT = 60


class AudioParser(BaseParser):
    name = "audio"

    def supported_extensions(self) -> list[str]:
        return SUPPORTED_EXTENSIONS

    def parse_file(self, path: Path, metadata: dict[str, Any] = {}, **kwargs) -> list[Part]:
        if not settings.whisper_host:
            raise FallbackError(
                "Audio transcription is not configured",
                parser_name=self.name,
                code="service_not_configured",
                detail="Set WHISPER_HOST to enable audio transcription.",
            )

        content = self.recognize_speech(path)
        metadata = metadata.copy()
        metadata["md_source_map"] = [0, content.count("\n") + 1]
        return [TextPart(content=content, metadata=metadata)]

    def recognize_speech(self, path: Path) -> str:
        params = {
            "encode": "true",
            "task": "transcribe",
            "vad_filter": "true",
            "word_timestamps": "true",
            "output": "txt",
        }
        cache = get_sync_application_cache()
        return cache.get_or_compute(
            namespace=NAMESPACE_REMOTE_PARSER,
            key_data={
                "parser": self.name,
                "file_hash": _file_sha256(path),
                "endpoint": settings.whisper_host,
                "params": params,
            },
            compute=lambda: self._recognize_speech_uncached(path, params),
            policy=application_cache_policy(NAMESPACE_REMOTE_PARSER),
            should_cache=lambda value: bool(value),
        )

    def _recognize_speech_uncached(self, path: Path, params: dict[str, str]) -> str:
        headers = {
            "Accept": "application/json",
        }

        # TODO: extract media metadata by using exiftool

        # Server: https://github.com/ahmetoner/whisper-asr-webservice
        try:
            with open(str(path), "rb") as audio_file:
                response = requests.post(
                    settings.whisper_host + "/asr",
                    params=params,
                    files={"audio_file": audio_file},
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                return response.text
        except requests.exceptions.RequestException as e:
            raise ParserError(
                "Audio transcription request failed",
                parser_name=self.name,
                code="service_unreachable",
                detail=str(e),
            ) from e


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
