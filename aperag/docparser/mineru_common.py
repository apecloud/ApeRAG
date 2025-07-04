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

import json
import logging
from hashlib import md5
from pathlib import Path
from typing import Any

from aperag.docparser.base import (
    AssetBinPart,
    ImagePart,
    MarkdownPart,
    Part,
    TextPart,
    TitlePart,
)
from aperag.docparser.utils import asset_bin_part_to_url, extension_to_mime_type

logger = logging.getLogger(__name__)


def to_md_part(parts: list[Part], metadata: dict[str, Any]) -> MarkdownPart:
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
    # Data from doc-ray has this field
    merged = block.get("merged_text", None)
    if merged is not None:
        return merged
    return _merge_para_with_text_impl(block)


def _merge_para_with_text_impl(para_block: dict) -> str:
    display_left_delimiter = "$$"
    display_right_delimiter = "$$"
    inline_left_delimiter = "$"
    inline_right_delimiter = "$"

    para_text = ""
    for line in para_block["lines"]:
        for j, span in enumerate(line["spans"]):
            span_type = span["type"]
            content = ""
            if span_type == ContentType.Text:
                content = span["content"]
            elif span_type == ContentType.InlineEquation:
                content = f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
            elif span_type == ContentType.InterlineEquation:
                content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
            # content = content.strip()
            if content:
                if span_type in [ContentType.Text, ContentType.InlineEquation]:
                    if j == len(line["spans"]) - 1:
                        para_text += content
                    else:
                        para_text += f"{content} "
                elif span_type == ContentType.InterlineEquation:
                    para_text += content
    return para_text


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
    Text = "text"
    InterlineEquation = "interline_equation"
    InlineEquation = "inline_equation"


def convert_para(
    image_dir: Path,
    para_block: dict[str, Any],
    page_idx: int,
    metadata: dict[str, Any],
) -> list[Part]:
    para_type = para_block["type"]
    bbox = para_block.get("bbox", (0, 0, 0, 0))
    metadata.update(
        {
            "pdf_source_map": [
                {
                    "page_idx": page_idx,
                    "bbox": tuple(bbox),
                }
            ],
            "para_type": str(para_type),
        }
    )

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
