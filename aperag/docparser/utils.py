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

import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote_plus

from aperag.docparser.base import AssetBinPart


def get_soffice_cmd() -> str | None:
    return shutil.which("soffice")


def convert_office_doc(input_path: Path, output_dir: Path, target_format: str) -> Path:
    soffice_cmd = get_soffice_cmd()
    if soffice_cmd is None:
        raise RuntimeError("soffice command not found")

    if not input_path.exists:
        raise FileNotFoundError(f"input file {input_path} not exist")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        soffice_cmd,
        "--headless",
        "--norestore",
        "--convert-to",
        target_format,
        "--outdir",
        str(output_dir),
        str(input_path),
    ]

    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(
            f'convert failed, cmd: "{" ".join(cmd)}", output: {process.stdout.decode()}, error: {process.stderr.decode()}'
        )

    return output_dir / f"{input_path.stem}.{target_format}"


def asset_bin_part_to_url(part: AssetBinPart) -> str:
    url = f"asset://{part.asset_id}"
    if part.mime_type:
        url += f"?mime_type={quote_plus(part.mime_type)}"
    return url


def extension_to_mime_type(extension: str) -> str:
    extension = extension.lower()
    if extension == ".pdf":
        return "application/pdf"
    elif extension == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif extension == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif extension == ".pptx":
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    elif extension == ".epub":
        return "application/epub+zip"
    elif extension == ".jpg" or extension == ".jpeg":
        return "image/jpeg"
    elif extension == ".png":
        return "image/png"
    elif extension == ".gif":
        return "image/gif"
    elif extension == ".svg":
        return "image/svg+xml"
    elif extension == ".webp":
        return "image/webp"
    else:
        return "application/octet-stream"
