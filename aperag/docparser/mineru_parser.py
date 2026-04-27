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
import io
import logging
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

from aperag.cache import NAMESPACE_REMOTE_PARSER, application_cache_policy, get_sync_application_cache
from aperag.docparser.base import (
    AssetBinPart,
    BaseParser,
    CodePart,
    FallbackError,
    ImagePart,
    MarkdownPart,
    MediaPart,
    ParserError,
    Part,
    PdfPart,
    TextPart,
    TitlePart,
)
from aperag.docparser.mineru_common import middle_json_to_parts, to_md_part

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
]

API_HOST = "https://mineru.net"
REQUEST_TIMEOUT = 30
MAX_POLL_SECONDS = 300


class MinerUParser(BaseParser):
    name = "mineru"

    def __init__(self, api_token: str = None, **kwargs):
        super().__init__(**kwargs)
        self.api_token = api_token

    def supported_extensions(self) -> list[str]:
        return SUPPORTED_EXTENSIONS

    def parse_file(self, path: Path, metadata: dict[str, Any], **kwargs) -> list[Part]:
        if not self.api_token:
            raise FallbackError(
                "MinerU is enabled but no API token is configured",
                parser_name=self.name,
                code="missing_configuration",
                detail="Set mineru_api_token to enable MinerU enhancement.",
            )

        token_scope = hashlib.sha256(self.api_token.encode("utf-8")).hexdigest()
        cache = get_sync_application_cache()
        payload = cache.get_or_compute(
            namespace=NAMESPACE_REMOTE_PARSER,
            key_data={
                "parser": self.name,
                "file_hash": _file_sha256(path),
                "endpoint": API_HOST,
                "model_version": "v2",
                "metadata": metadata,
                "token_scope": token_scope,
            },
            compute=lambda: _parts_to_cache_payload(self._parse_file_uncached(path, metadata)),
            policy=application_cache_policy(NAMESPACE_REMOTE_PARSER),
            should_cache=lambda value: bool(value),
        )
        return _parts_from_cache_payload(payload)

    def _parse_file_uncached(self, path: Path, metadata: dict[str, Any]) -> list[Part]:

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        # 1. Get upload URL
        upload_url_payload = {
            "language": "auto",
            "files": [{"name": path.name, "is_ocr": True}],
            "model_version": "v2",
        }
        try:
            resp = requests.post(
                f"{API_HOST}/api/v4/file-urls/batch",
                headers=headers,
                json=upload_url_payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            upload_data = resp.json()
            if upload_data.get("code") != 0:
                raise FallbackError(
                    "MinerU could not allocate an upload slot",
                    parser_name=self.name,
                    code="service_rejected_request",
                    detail=upload_data.get("msg"),
                )

            batch_id = upload_data["data"]["batch_id"]
            file_url = upload_data["data"]["file_urls"][0]
            logger.info(f"Got Mineru upload URL for {path.name}, batch_id: {batch_id}")

        except requests.exceptions.RequestException as e:
            logger.exception("Failed to get Mineru upload URL")
            raise FallbackError(
                "MinerU upload initialization failed",
                parser_name=self.name,
                code="service_unreachable",
                detail=str(e),
            ) from e

        # 2. Upload file
        try:
            with open(path, "rb") as f:
                upload_resp = requests.put(file_url, data=f, timeout=REQUEST_TIMEOUT)
                upload_resp.raise_for_status()
            logger.info(f"Successfully uploaded {path.name} to Mineru.")
        except requests.exceptions.RequestException as e:
            logger.exception(f"Failed to upload file to Mineru: {path.name}")
            raise FallbackError(
                "MinerU file upload failed",
                parser_name=self.name,
                code="service_unreachable",
                detail=str(e),
            ) from e

        # 3. Poll for result
        start_time = time.monotonic()
        while True:
            if time.monotonic() - start_time > MAX_POLL_SECONDS:
                raise FallbackError(
                    "MinerU parsing timed out",
                    parser_name=self.name,
                    code="timeout",
                    detail=f"Timed out after {MAX_POLL_SECONDS} seconds. Falling back to local parser.",
                )
            try:
                time.sleep(5)  # Poll every 5 seconds
                status_resp = requests.get(
                    f"{API_HOST}/api/v4/extract-results/batch/{batch_id}",
                    headers={"Authorization": f"Bearer {self.api_token}"},
                    timeout=REQUEST_TIMEOUT,
                )
                status_resp.raise_for_status()
                status_data = status_resp.json()

                if status_data.get("code") != 0:
                    logger.warning(f"Polling failed for batch {batch_id}: {status_data.get('msg')}")
                    continue

                extract_result = status_data.get("data", {}).get("extract_result", [])
                if not extract_result:
                    continue

                task_status = extract_result[0]
                state = task_status.get("state")
                logger.info(f"Mineru job {batch_id} status: {state}")

                if state == "done":
                    zip_url = task_status.get("full_zip_url")
                    if not zip_url:
                        raise RuntimeError("Mineru job completed but no zip_url found.")
                    logger.info(f"Mineru job {batch_id} completed. Downloading from {zip_url}")
                    is_pdf_input = path.suffix.lower() == ".pdf"
                    return self._download_and_process_zip(zip_url, metadata, is_pdf_input)
                elif state == "failed":
                    error_message = task_status.get("err_msg", "Unknown error")
                    # "number of pages exceeds limit" or "file size exceeds limit"
                    if "exceeds limit" in error_message:
                        raise FallbackError(
                            "MinerU rejected the file because it exceeds service limits",
                            parser_name=self.name,
                            code="service_limit_exceeded",
                            detail=error_message,
                        )
                    raise FallbackError(
                        "MinerU parsing failed and the system will fall back to the local parser",
                        parser_name=self.name,
                        code="service_failed",
                        detail=error_message,
                    )

            except requests.exceptions.RequestException as e:
                logger.warning(f"Polling request failed for batch {batch_id}: {e}")
                raise FallbackError(
                    "MinerU status polling failed",
                    parser_name=self.name,
                    code="service_unreachable",
                    detail=str(e),
                ) from e

    def _download_and_process_zip(self, zip_url: str, metadata: dict[str, Any], is_pdf_input: bool) -> list[Part]:
        temp_dir_obj = None
        try:
            response = requests.get(zip_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir_obj.name)
            image_dir = temp_dir_path / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            middle_json_content = None
            pdf_part = None

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                for filename in z.namelist():
                    if filename.startswith("images/"):
                        z.extract(filename, temp_dir_path)
                    elif filename == "layout.json":
                        middle_json_content = z.read(filename).decode("utf-8")
                    elif not is_pdf_input and filename.endswith("_origin.pdf"):
                        pdf_data = z.read(filename)
                        pdf_part = PdfPart(data=pdf_data, metadata=metadata.copy())

            if not middle_json_content:
                raise FallbackError(
                    "MinerU response did not contain layout.json",
                    parser_name=self.name,
                    code="invalid_response",
                    detail="Falling back to the local parser.",
                )

            parts = middle_json_to_parts(image_dir, middle_json_content, metadata.copy())
            if not parts:
                return []

            md_part = to_md_part(parts, metadata.copy())
            final_parts = [md_part] + parts
            if pdf_part:
                final_parts.append(pdf_part)
            return final_parts

        except requests.exceptions.RequestException as e:
            logger.exception(f"Failed to download result zip from {zip_url}")
            raise FallbackError(
                "MinerU result download failed",
                parser_name=self.name,
                code="service_unreachable",
                detail=str(e),
            ) from e
        except FallbackError:
            raise
        except Exception as e:
            raise ParserError(
                "MinerU returned an invalid parsing result",
                parser_name=self.name,
                code="invalid_response",
                detail=str(e),
            ) from e
        finally:
            if temp_dir_obj:
                temp_dir_obj.cleanup()


_PART_TYPES = {
    "Part": Part,
    "MarkdownPart": MarkdownPart,
    "PdfPart": PdfPart,
    "AssetBinPart": AssetBinPart,
    "TextPart": TextPart,
    "TitlePart": TitlePart,
    "CodePart": CodePart,
    "MediaPart": MediaPart,
    "ImagePart": ImagePart,
}


def _parts_to_cache_payload(parts: list[Part]) -> list[dict[str, Any]]:
    return [{"type": part.__class__.__name__, "data": _encode_bytes(part.model_dump(mode="python"))} for part in parts]


def _parts_from_cache_payload(payload: list[dict[str, Any]]) -> list[Part]:
    parts: list[Part] = []
    for item in payload:
        model = _PART_TYPES.get(item.get("type"), Part)
        parts.append(model.model_validate(_decode_bytes(item.get("data", {}))))
    return parts


def _encode_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {key: _encode_bytes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_bytes(item) for item in value]
    return value


def _decode_bytes(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__bytes_b64__"}:
            return base64.b64decode(value["__bytes_b64__"])
        return {key: _decode_bytes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_bytes(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- For testing ---
if __name__ == "__main__":
    import os
    import sys

    from dotenv import load_dotenv

    # Add project root to sys.path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Load .env file from project root
    load_dotenv(dotenv_path=project_root / ".env")

    logging.basicConfig(level=logging.INFO)

    api_token = os.getenv("MINERU_API_TOKEN")
    if not api_token:
        print("Error: MINERU_API_TOKEN environment variable not set.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <file_path>")
        sys.exit(1)

    file_to_parse = Path(sys.argv[1])
    if not file_to_parse.exists():
        print(f"Error: File not found at {file_to_parse}")
        sys.exit(1)

    print(f"Testing MineruApiParser with file: {file_to_parse}")
    parser = MinerUParser(api_token=api_token)
    try:
        parsed_parts = parser.parse_file(file_to_parse, {})
        print("\n--- Parsing Result ---")
        for i, part in enumerate(parsed_parts):
            print(f"Part {i + 1}: {part.__class__.__name__}")
            print(f"  Content: {part.content[:100] if part.content else 'N/A'}...")
            print(f"  Metadata: {part.metadata}")
        print("\n--- Test Completed ---")
    except (FallbackError, RuntimeError) as e:
        print("\n--- Test Failed ---")
        print(f"An error occurred: {e}")
        print("---------------------")
