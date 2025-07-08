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
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from aperag.config import settings
from aperag.docparser.base import (
    BaseParser,
    FallbackError,
    Part,
    PdfPart,
)
from aperag.docparser.mineru_common import middle_json_to_parts, to_md_part

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
]


class DocRayParser(BaseParser):
    name = "docray"

    def supported_extensions(self) -> list[str]:
        return SUPPORTED_EXTENSIONS

    def parse_file(self, path: Path, metadata: dict[str, Any], **kwargs) -> list[Part]:
        if not settings.docray_host:
            raise FallbackError("DOCRAY_HOST is not set")

        job_id = None
        temp_dir_obj = None
        try:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir_obj.name)

            # Submit file to doc-ray
            with open(path, "rb") as f:
                files = {"file": (path.name, f)}
                response = requests.post(f"{settings.docray_host}/submit", files=files)
                response.raise_for_status()
                submit_response = response.json()
                job_id = submit_response["job_id"]
                logger.info(f"Submitted file {path.name} to DocRay, job_id: {job_id}")

            # Polling the processing status
            while True:
                time.sleep(5)  # Poll every 5 second
                status_response: dict = requests.get(f"{settings.docray_host}/status/{job_id}").json()
                status = status_response["status"]
                logger.info(f"DocRay job {job_id} status: {status}")

                if status == "completed":
                    break
                elif status == "failed":
                    error_message = status_response.get("error", "Unknown error")
                    raise RuntimeError(f"DocRay parsing failed for job {job_id}: {error_message}")
                elif status not in ["processing"]:
                    raise RuntimeError(f"Unexpected DocRay job status for {job_id}: {status}")

            # Get the result
            result_response = requests.get(f"{settings.docray_host}/result/{job_id}").json()
            result = result_response["result"]
            middle_json = result["middle_json"]
            images_data = result.get("images", {})

            # Dump image files into temp dir
            for img_name, img_base64 in images_data.items():
                img_file_path = temp_dir_path / str(img_name)

                # Ensure the resolved path is within the temporary directory.
                resolved_img_file_path = img_file_path.resolve()
                resolved_temp_dir_path = temp_dir_path.resolve()
                if not resolved_img_file_path.is_relative_to(resolved_temp_dir_path):
                    logger.error(
                        f"Security: Prevented writing image to an unintended path. "
                        f"File name: '{img_name}' "
                        f"Attempted path: '{resolved_img_file_path}', "
                        f"Temp dir: '{resolved_temp_dir_path}'"
                    )
                    continue

                img_file_path.parent.mkdir(parents=True, exist_ok=True)
                img_data = base64.b64decode(img_base64)
                with open(img_file_path, "wb") as f_img:
                    f_img.write(img_data)

            if metadata is None:
                metadata = {}
            parts = middle_json_to_parts(temp_dir_path / "images", middle_json, metadata)
            if not parts:
                return []

            pdf_data = result.get("pdf_data", None)
            if pdf_data:
                pdf_part = PdfPart(data=base64.b64decode(pdf_data), metadata=metadata.copy())
                parts.append(pdf_part)

            md_part = to_md_part(parts, metadata.copy())
            return [md_part] + parts

        except requests.exceptions.RequestException:
            logger.exception("DocRay API request failed")
            raise
        except Exception:
            logger.exception("DocRay parsing failed")
            raise
        finally:
            # Delete the job in doc-ray to release resources
            if job_id:
                try:
                    requests.delete(f"{settings.docray_host}/result/{job_id}")
                    logger.info(f"Deleted DocRay job {job_id}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to delete DocRay job {job_id}: {e}")
            if temp_dir_obj:
                temp_dir_obj.cleanup()
