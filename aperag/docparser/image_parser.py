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

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from aperag.cache import NAMESPACE_REMOTE_PARSER, application_cache_policy, get_sync_application_cache
from aperag.config import settings
from aperag.docparser.base import BaseParser, FallbackError, ParserError, Part, TextPart

SUPPORTED_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
]

REQUEST_TIMEOUT = 15


class ImageParser(BaseParser):
    name = "image"

    def supported_extensions(self) -> list[str]:
        return SUPPORTED_EXTENSIONS

    def parse_file(self, path: Path, metadata: dict[str, Any] = {}, **kwargs) -> list[Part]:
        if not settings.paddleocr_host:
            raise FallbackError(
                "Image OCR is not configured",
                parser_name=self.name,
                code="service_not_configured",
                detail="Set PADDLEOCR_HOST to enable image OCR.",
            )

        content = self.read_image_text(path)
        metadata = metadata.copy()
        metadata["md_source_map"] = [0, content.count("\n") + 1]
        return [TextPart(content=content, metadata=metadata)]

    def read_image_text(self, path: Path) -> str:
        cache = get_sync_application_cache()
        return cache.get_or_compute(
            namespace=NAMESPACE_REMOTE_PARSER,
            key_data={
                "parser": self.name,
                "file_hash": _file_sha256(path),
                "endpoint": settings.paddleocr_host,
                "request": "ocr_system",
            },
            compute=lambda: self._read_image_text_uncached(path),
            policy=application_cache_policy(NAMESPACE_REMOTE_PARSER),
            should_cache=lambda value: bool(value),
        )

    def _read_image_text_uncached(self, path: Path) -> str:
        def image_to_base64(image_path: str):
            with Image.open(image_path) as image:
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_byte = buffered.getvalue()
                img_base64 = base64.b64encode(img_byte)
                return img_base64.decode()

        data = {"images": [image_to_base64(str(path))]}
        headers = {"Content-type": "application/json"}
        url = settings.paddleocr_host + "/predict/ocr_system"
        try:
            r = requests.post(url=url, headers=headers, data=json.dumps(data), timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = json.loads(r.text)
        except requests.exceptions.RequestException as e:
            raise ParserError(
                "Image OCR request failed",
                parser_name=self.name,
                code="service_unreachable",
                detail=str(e),
            ) from e
        except json.JSONDecodeError as e:
            raise ParserError(
                "Image OCR returned invalid JSON",
                parser_name=self.name,
                code="invalid_response",
                detail=str(e),
            ) from e

        # TODO: extract image metadata by using exiftool

        texts = [item["text"] for group in data["results"] for item in group if "text" in item]
        res = ""
        for text in texts:
            res += text
        return res


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
