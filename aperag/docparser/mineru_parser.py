import json
import logging
import os
import tempfile
from hashlib import md5
from pathlib import Path
from typing import Any

import fitz
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.data.read_api import read_local_office
from magic_pdf.dict2md.ocr_mkcontent import merge_para_with_text
from magic_pdf.libs import config_reader
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.operators.pipes import PipeResult

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
from aperag.docparser.utils import asset_bin_part_to_url, extension_to_mime_type, get_soffice_cmd

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [
    ".pdf",
    # convert to .pdf first
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
]


class MinerUParser(BaseParser):
    def _set_config_path(self) -> bool:
        path = Path(os.environ.get("MINERU_CONFIG_JSON", "./magic-pdf.json"))
        if not path.exists():
            return False

        def _adjust_cache_path(cfg: dict, key: str, new_cache_dir: str):
            cache_dir_name = ".cache"
            orig = cfg.get(key, "")
            pos = orig.find(cache_dir_name)
            if pos == -1:
                return
            cfg[key] = str(Path(new_cache_dir + "/" + orig[pos + len(cache_dir_name) :]))

        # Adjust the model dir when running in docker.
        new_cache_dir = os.environ.get("MINERU_ADJUST_MODEL_CACHE_DIR", None)
        if new_cache_dir is not None:
            cfg = {}
            with open(str(path), "r") as f:
                cfg = json.loads(f.read())
            _adjust_cache_path(cfg, "models-dir", new_cache_dir)
            _adjust_cache_path(cfg, "layoutreader-model-dir", new_cache_dir)
            content = json.dumps(cfg)

            path = Path(tempfile.gettempdir()) / "magic-pdf-mod.json"

            should_write_file = True
            if path.exists():
                curr_content = ""
                with path.open("r") as f:
                    curr_content = f.read()
                should_write_file = content != curr_content
            if should_write_file:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as f:
                    f.write(content)

        config_reader.CONFIG_FILE_NAME = str(path.absolute())

        return True

    def supported_extensions(self) -> list[str]:
        return SUPPORTED_EXTENSIONS

    def parse_file(self, path: Path, metadata: dict[str, Any], **kwargs) -> list[Part]:
        extension = path.suffix.lower()
        if extension != ".pdf":
            if get_soffice_cmd() is None:
                raise FallbackError("soffice command not found")
        if not self._set_config_path():
            raise FallbackError("magic-pdf.json not found")

        temp_dir = os.environ.get("MINERU_TEMP_FILE_DIR", None)
        temp_dir_obj: tempfile.TemporaryDirectory | None = None
        if not temp_dir:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

        try:
            local_image_dir = os.path.join(temp_dir, "output/images")

            os.makedirs(local_image_dir, exist_ok=True)

            image_writer = FileBasedDataWriter(local_image_dir)

            parse_method = SupportedPdfParseMethod.OCR
            ds: PymuDocDataset = None
            if path.suffix == ".pdf":
                reader1 = FileBasedDataReader("")
                pdf_bytes = reader1.read(str(path))
                ds = PymuDocDataset(pdf_bytes)
                parse_method = ds.classify()
            else:
                # Note: this requires the "soffice" command to convert office docs into PDF.
                # The "soffice" command is part of LibreOffice, can be installed via:
                #   apt-get install libreoffice
                #   brew install libreoffice
                ds = read_local_office(str(path))[0]

                # TODO: save the converted pdf file

            pipe_result: PipeResult = None
            if parse_method == SupportedPdfParseMethod.OCR:
                result = ds.apply(doc_analyze, ocr=True)
                pipe_result = result.pipe_ocr_mode(image_writer)
            else:
                result = ds.apply(doc_analyze, ocr=False)
                pipe_result = result.pipe_txt_mode(image_writer)

            if hasattr(pipe_result, "_pipe_res"):
                adjust_title_level(path, pipe_result._pipe_res)

            if metadata is None:
                metadata = {}
            parts = middle_json_to_parts(Path(local_image_dir), pipe_result.get_middle_json(), metadata)
            if not parts:
                return []
            md_part = self.to_md_part(parts, metadata.copy())
            return [md_part] + parts
        except:
            logger.exception("MinerUParser failed")
            raise
        finally:
            if temp_dir_obj is not None:
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


def adjust_title_level(pdf_file: Path | None, pipe_res: dict):
    logger.info("Adjusting title level...")
    raw_text_blocks: dict[int, list[tuple[str, float]]] = {}
    if pdf_file is not None:
        raw_text_blocks = collect_all_text_blocks(pdf_file)
    title_blocks = []
    for page_num, page_info in enumerate(pipe_res.get("pdf_info", [])):
        paras_of_layout: list[dict[str, Any]] = page_info.get("para_blocks")
        if not paras_of_layout:
            continue
        for para_block in paras_of_layout:
            para_type = para_block["type"]
            if para_type != BlockType.Title:
                continue
            has_level = para_block.get("level", None)
            if has_level is not None:
                logger.info("MinerU has already set a title level; skipping adjustment.")
                return

            raw_text_map = {}
            for raw_text in raw_text_blocks.get(page_num, []):
                raw_text_map[raw_text[0]] = raw_text[1]

            font_size = None
            lines = para_block.get("lines", [])
            for line in lines:
                spans = line.get("spans", [])
                for span in spans:
                    content = span.get("content", "").strip()
                    if content in raw_text_map:
                        font_size = raw_text_map[content]

            # If the font size cannot be obtained directly from the PDF document,
            # calculate an approximate font size based on the bounding box height.
            if font_size is None:
                bbox = para_block.get("bbox", None)
                if bbox is not None:
                    height = bbox[3] - bbox[1]
                    lines = para_block.get("lines", None)
                    if lines is not None and len(lines) > 1:
                        height = height / len(lines)
                    # NOTE: This formula is derived from simple observation
                    # and may not be applicable to all situations.
                    font_size = height * 0.78

            if font_size is None:
                continue

            title_blocks.append(
                (
                    font_size,
                    para_block,
                )
            )

    if len(title_blocks) == 0:
        return

    title_blocks.sort(key=lambda x: x[0], reverse=True)
    level = 1
    prev_font_size = None
    delta = 0.2
    max_level = 8
    for font_size, para_block in title_blocks:
        if prev_font_size is not None and prev_font_size - font_size > delta:
            level += 1
            if level > max_level:
                level = max_level
        para_block["level"] = level
        prev_font_size = font_size


def collect_all_text_blocks(pdf_path: Path) -> dict[int, list[tuple[str, float]]]:
    try:
        with fitz.open(pdf_path) as doc:
            if not doc.is_pdf:
                return {}

            ret = {}
            for page_num, page in enumerate(doc):
                try:
                    # Extract text using 'dict' mode, which returns a dictionary structure
                    # containing detailed information: page -> block -> line -> span.
                    # Each span includes font size, content, etc.
                    page_data = page.get_text("dict")

                    blocks = page_data.get("blocks", [])
                    if not blocks:
                        continue

                    texts = []
                    # Iterate over all blocks in the page
                    for block in blocks:
                        # Check if the block is a text block (type 0) and contains line information
                        if block.get("type") == 0 and "lines" in block:
                            lines = block.get("lines", [])
                            # Iterate over all lines in the block
                            for line in lines:
                                spans = line.get("spans", [])
                                # Iterate over all spans in the line
                                for span in spans:
                                    font_size = span.get("size", 1.0)
                                    text_content = span.get("text", "").strip()
                                    if text_content:
                                        texts.append(
                                            (
                                                text_content,
                                                font_size,
                                            )
                                        )

                    ret[page_num] = texts
                except Exception:
                    logger.exception(f"collect_all_text_blocks error processing page {page_num + 1}")

            return ret
    except Exception:
        logger.exception("collect_all_text_blocks failed")
        return {}


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
