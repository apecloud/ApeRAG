import base64
import json
import logging
import tempfile
import time
from hashlib import md5
from pathlib import Path
from typing import Any

import requests

from aperag.docparser.base import (
    AssetBinPart,
    BaseParser,
    FallbackError,
    ImagePart,
    MarkdownPart,
    Part,
    TextPart,
    TitlePart,
)
from aperag.docparser.utils import asset_bin_part_to_url, extension_to_mime_type
from config import settings

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
        if not settings.DOCRAY_HOST:
            raise FallbackError("DOCRAY_HOST is not set")

        job_id = None
        temp_dir_obj = None
        try:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir_obj.name)

            # Submit file to doc-ray
            with open(path, "rb") as f:
                files = {"file": (path.name, f)}
                response = requests.post(f"{settings.DOCRAY_HOST}/submit", files=files)
                response.raise_for_status()
                submit_response = response.json()
                job_id = submit_response["job_id"]
                logger.info(f"Submitted file {path.name} to DocRay, job_id: {job_id}")

            # Polling the processing status
            while True:
                time.sleep(5)  # Poll every 5 second
                status_response: dict = requests.get(f"{settings.DOCRAY_HOST}/status/{job_id}").json()
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
            result_response = requests.get(f"{settings.DOCRAY_HOST}/result/{job_id}").json()
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
            md_part = self.to_md_part(parts, metadata.copy())
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
                    requests.delete(f"{settings.DOCRAY_HOST}/result/{job_id}")
                    logger.info(f"Deleted DocRay job {job_id}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to delete DocRay job {job_id}: {e}")
            if temp_dir_obj:
                temp_dir_obj.cleanup()

    def to_md_part(self, parts: list[Part], metadata: dict[str, Any]) -> MarkdownPart:
        pos = 0
        md = ""
        for part in parts:
            if isinstance(part, AssetBinPart):
                continue
            if part.content:
                content = part.content.rstrip() + "\n"
                lines = content.count("\n")
                part.metadata["md_source_map"] = [pos, pos + lines]
                md += content + "\n"
                pos += lines + 1

        return MarkdownPart(
            metadata=metadata,
            markdown=md,
        )


def middle_json_to_parts(image_dir: Path, middle_json: str, metadata: dict[str, Any]) -> list[Part]:
    result: list[Part] = []
    middle: dict[str, Any] = json.loads(middle_json)
    for page_info in middle.get("pdf_info", []):
        paras_of_layout: list[dict[str, Any]] = page_info.get("para_blocks")
        page_idx: int = page_info.get("page_idx")
        if not paras_of_layout:
            continue
        for para_block in paras_of_layout:
            parts = convert_para(image_dir, para_block, page_idx, metadata.copy())
            result.extend(parts)
    return result


def merge_para_with_text(block: dict[str, Any]) -> str:
    return block.get("merged_text", "")


class BlockType:
    Text = "text"
    List = "list"
    Index = "index"
    Title = "title"
    InterlineEquation = "interline_equation"
    Image = "image"
    ImageBody = "image_body"
    ImageCaption = "image_caption"
    ImageFootnote = "image_footnote"
    Table = "table"
    TableBody = "table_body"
    TableCaption = "table_caption"
    TableFootnote = "table_footnote"


class ContentType:
    Image = "image"
    Table = "table"


def convert_para(
    image_dir: Path,
    para_block: dict[str, Any],
    page_idx: int,
    metadata: dict[str, Any],
) -> list[Part]:
    para_type = para_block["type"]
    bbox = para_block.get("bbox", (0, 0, 0, 0))
    metadata.update({
        "pdf_source_map": [
            {
                "page_idx": page_idx,
                "bbox": tuple(bbox),
            }
        ],
        "para_type": str(para_type),
    })

    if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
        return [
            TextPart(
                content=merge_para_with_text(para_block),
                metadata=metadata,
            )
        ]
    elif para_type == BlockType.Title:
        title_level = para_block.get("level", 1)
        return [
            TitlePart(
                content=f"{'#' * title_level} {merge_para_with_text(para_block)}",
                metadata=metadata,
                level=title_level,
            )
        ]
    elif para_type == BlockType.InterlineEquation:
        return [
            TextPart(
                content=merge_para_with_text(para_block),
                metadata=metadata,
            )
        ]
    elif para_type == BlockType.Image:
        return _convert_image_para(image_dir, para_block, metadata)
    elif para_type == BlockType.Table:
        return _convert_table_para(image_dir, para_block, metadata)

    return []


def _convert_image_para(image_dir: Path, para_block: dict[str, Any], metadata: dict[str, Any]) -> list[Part]:
    img_path = None
    text = ""
    for block in para_block["blocks"]:
        if block["type"] == BlockType.ImageBody:
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["type"] == ContentType.Image:
                        if span.get("image_path", ""):
                            img_path = span["image_path"]
        if block["type"] == BlockType.ImageCaption:
            text += f"[ImageCaption: {merge_para_with_text(block)}]\n"
        if block["type"] == BlockType.ImageFootnote:
            text += f"[ImageFootnote: {merge_para_with_text(block)}]\n"

    if len(text) == 0:
        return []

    img_data = None
    try:
        img_full_path = image_dir / img_path
        with open(img_full_path, "rb") as f:
            img_data = f.read()
    except Exception:
        logger.exception(f"failed to read image {img_full_path}")

    if img_data is None:
        return [TextPart(content=text, metadata=metadata)]

    asset_id = md5(img_data).hexdigest()
    mime_type = extension_to_mime_type(Path(img_path).suffix)
    asset_bin_part = AssetBinPart(
        asset_id=asset_id,
        data=img_data,
        metadata=metadata,
        mime_type=mime_type,
    )

    asset_url = asset_bin_part_to_url(asset_bin_part)
    text = f"![{img_path}]({asset_url})\n" + text

    img_part = ImagePart(
        content=text,
        metadata=metadata,
        url=asset_url,
    )
    return [asset_bin_part, img_part]


def _convert_table_para(image_dir: Path, para_block: dict[str, Any], metadata: dict[str, Any]) -> list[Part]:
    img_path = None
    text = ""
    for block in para_block["blocks"]:
        if block["type"] == BlockType.TableBody:
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["type"] == ContentType.Table:
                        table_body = ""
                        table_format = None
                        if span.get("latex", ""):
                            table_body = f"\n\n$\n {span['latex']}\n$\n\n"
                            table_format = "latex"
                        elif span.get("html", ""):
                            table_body = f"\n\n{span['html']}\n\n"
                            table_format = "html"

                        if span.get("image_path", ""):
                            img_path = span["image_path"]

                        if len(table_body) > 0:
                            text = f"Table ({table_format}):\n{table_body}\n"

        if block["type"] == BlockType.TableCaption:
            text += f"[TableCaption: {merge_para_with_text(block)}]\n"
        if block["type"] == BlockType.TableFootnote:
            text += f"[TableFootnote: {merge_para_with_text(block)}]\n"

    if len(text) == 0:
        return []

    img_data = None
    if img_path:
        try:
            img_full_path = image_dir / img_path
            with open(img_full_path, "rb") as f:
                img_data = f.read()
        except Exception:
            logger.exception(f"failed to read image {img_full_path}")

    if img_data is None:
        metadata["table_format"] = table_format
        return [TextPart(content=text, metadata=metadata)]

    asset_id = md5(img_data).hexdigest()
    mime_type = extension_to_mime_type(Path(img_path).suffix)
    asset_bin_part = AssetBinPart(
        asset_id=asset_id,
        data=img_data,
        metadata=metadata,
        mime_type=mime_type,
    )

    asset_url = asset_bin_part_to_url(asset_bin_part)
    text = f"![{img_path}]({asset_url})\n" + text

    img_part = ImagePart(
        content=text,
        metadata=metadata,
        url=asset_url,
    )
    return [asset_bin_part, img_part]
