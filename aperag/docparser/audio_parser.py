from pathlib import Path
from typing import Any

import requests

from aperag.docparser.base import BaseParser, FallbackError, Part, TextPart
from config import settings

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


class AudioParser(BaseParser):
    name = "audio"

    def supported_extensions(self) -> list[str]:
        return SUPPORTED_EXTENSIONS

    def parse_file(self, path: Path, metadata: dict[str, Any] = {}, **kwargs) -> list[Part]:
        if not settings.WHISPER_HOST:
            raise FallbackError("WHISPER_HOST is not set")

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

        files = {"audio_file": open(str(path), "rb")}

        headers = {
            "Accept": "application/json",
        }

        # TODO: extract media metadata by using exiftool

        # Server: https://github.com/ahmetoner/whisper-asr-webservice
        response = requests.post(settings.WHISPER_HOST + "/asr", params=params, files=files, headers=headers)
        return response.text
