import base64
import inspect
import logging
import re
from dataclasses import dataclass
from hashlib import md5
from typing import Any, Pattern

from markdown_it import MarkdownIt
from markdown_it.token import Token

from aperag.docparser.base import AssetBinPart, CodePart, ImagePart, MarkdownPart, Part, TextPart, TitlePart
from aperag.docparser.utils import asset_bin_part_to_url

logger = logging.getLogger(__name__)

DATA_URI_PATTERN: Pattern = re.compile(r"!\[.*?\]\(\s*(data:.+?;base64,.+?)(?:\s+\"(.*?)\")?\)")


def parse_md(input_md: str, metadata: dict[str, Any]) -> list[Part]:
    input_md, asset_bin_parts = extract_data_uri(input_md, metadata)
    md_part = MarkdownPart(markdown=input_md, metadata=metadata)

    md = MarkdownIt("gfm-like", options_update={"inline_definitions": True})
    tokens = md.parse(input_md)
    converter = PartConverter()
    parts = converter.convert_all(tokens, metadata)

    return [md_part] + asset_bin_parts + parts


def extract_data_uri(text: str, metadata: dict[str, Any]) -> tuple[str, list[Part]]:
    asset_bin_parts: list[Part] = []
    for match in DATA_URI_PATTERN.finditer(text):
        data_uri = match.group(1)

        try:
            mime_type, encoded_data = data_uri.split("base64,")
            mime_type = mime_type[5:-1]  # Remove 'data:' and the trailing ';'
            binary_data = base64.b64decode(encoded_data)

            asset_id = md5(binary_data).hexdigest()
            asset_bin_part = AssetBinPart(
                asset_id=asset_id,
                data=binary_data,
                mime_type=mime_type,
                metadata=metadata,
            )
            asset_bin_parts.append(asset_bin_part)

            asset_url = asset_bin_part_to_url(asset_bin_part)
            text = text.replace(data_uri, asset_url)
        except Exception as e:
            logger.warning(f"Error processing data URI: {e}")

    return text, asset_bin_parts


class PartConverter:
    @dataclass
    class Context:
        nesting: int = 0
        ordinal: int = 1
        pause_extraction: bool = False

    class Nester:
        def __init__(self, ctx: "PartConverter.Context"):
            self.ctx = ctx

        def __enter__(self):
            self.ctx.nesting += 1
            return self.ctx

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.ctx.nesting -= 1
            return False

    class OrderedListNester:
        def __init__(self, ctx: "PartConverter.Context"):
            self.ctx = ctx
            self.old_ordinal = self.ctx.ordinal

        def __enter__(self):
            self.ctx.ordinal = 1
            self.ctx.nesting += 1
            return self.ctx

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.ctx.ordinal = self.old_ordinal
            self.ctx.nesting -= 1
            return False

    class PauseExtraction:
        def __init__(self, ctx: "PartConverter.Context"):
            self.ctx = ctx
            self.old_val = ctx.pause_extraction

        def __enter__(self):
            self.ctx.pause_extraction = True
            return self.ctx

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.ctx.pause_extraction = self.old_val
            return False

    def __init__(self):
        prefix = "_convert_"
        self.handlers = {
            k[len(prefix) :]: v
            for k, v in inspect.getmembers(self, predicate=inspect.ismethod)
            if k.startswith(prefix) and len(k) > len(prefix)
        }

    def convert(self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]) -> tuple[list[Part], int]:
        if idx >= len(tokens):
            return [], idx
        token = tokens[idx]
        handler = self.handlers.get(token.type)
        if handler:
            return handler(ctx, tokens, idx, metadata.copy())
        logger.warning(f"Unhandled token type: {token}")
        if token.content:
            meta = metadata.copy()
            if token.map:
                meta["md_source_map"] = token.map
            return [TextPart(content=token.content, metadata=meta)], idx + 1
        return [], idx + 1

    def convert_all(self, tokens: list[Token], metadata: dict[str, Any]) -> list[Part]:
        if metadata is None:
            metadata = {}
        result: list[Part] = []
        ctx = self.Context()
        pos = 0
        while pos < len(tokens):
            parts, pos = self.convert(ctx, tokens, pos, metadata)
            result.extend(parts)
        return result

    def convert_until_close(
        self, ctx: Context, close_ttype: str, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        result: list[Part] = []
        pos = idx
        while pos < len(tokens):
            next_token = tokens[pos]
            if next_token.type == close_ttype:
                pos = pos + 1
                break
            parts, pos = self.convert(ctx, tokens, pos, metadata)
            result.extend(parts)
        return result, pos

    def _extract_image_parts(self, ctx: Context, tokens: list[Token], metadata: dict[str, Any]) -> list[Part]:
        if ctx.pause_extraction:
            return []
        result: list[Part] = []
        for token in tokens:
            if not token.children:
                continue
            for child in token.children:
                if child.children:
                    result.extend(self._extract_image_parts(ctx, child.children, metadata))
                if child.type != "image":
                    continue
                if child.meta.get("_extracted_image_token", False):
                    continue
                child.meta["_extracted_image_token"] = True
                meta = metadata.copy()
                meta["md_source_map"] = child.map
                if child.type == "image":
                    img_part = ImagePart(
                        metadata=meta,
                        url=child.attrs.get("src", ""),
                        alt_text=child.content,
                        title=child.attrs.get("title", None),
                    )
                    result.append(img_part)
        return result

    # ======================================================
    # handlers
    # ======================================================

    def _convert_blockquote_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        with self.Nester(ctx):
            parts, pos = self.convert_until_close(ctx, "blockquote_close", tokens, idx + 1, metadata)

        token = tokens[idx]
        for part in parts:
            if not isinstance(
                part,
                (
                    TextPart,
                    TitlePart,
                ),
            ):
                continue
            if part.content is not None:
                lines = part.content.split("\n")
                # Add "> " prefix to each line
                part.content = "\n".join([token.markup + " " + line for line in lines])

        return parts, pos

    @staticmethod
    def _to_code_content(code: str, lang: str | None = None) -> str:
        backticks = "```"
        for i in range(10):
            if backticks not in code:
                break
            backticks += "`"
        code = code.strip()
        if lang:
            return f"{backticks}{lang}\n{code}\n{backticks}"
        return f"{backticks}\n{code}\n{backticks}"

    def _convert_code_block(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        token = tokens[idx]
        metadata["md_source_map"] = token.map
        metadata["md_nesting"] = ctx.nesting
        code = self._to_code_content(token.content, None)
        return [CodePart(content=code, metadata=metadata)], idx + 1

    def _convert_fence(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        token = tokens[idx]
        metadata["md_source_map"] = token.map
        metadata["md_nesting"] = ctx.nesting
        lang = None
        if token.info:
            lang = token.info
            metadata["code_lang"] = lang
        code = self._to_code_content(token.content, lang)
        return [CodePart(content=code, metadata=metadata, lang=lang)], idx + 1

    def _convert_heading_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        parts, pos = self.convert_until_close(ctx, "heading_close", tokens, idx + 1, metadata)
        text = ""
        for part in parts:
            if part.content is not None:
                text += part.content
        token = tokens[idx]
        metadata["md_source_map"] = token.map
        metadata["md_nesting"] = ctx.nesting
        if token.markup in ["=", "-"]:
            # It's a lheading
            if token.markup == "=":
                level = 1
            else:
                level = 2
        else:
            level = len(token.markup)  # Count how many "#"
        title = ("#" * level) + " " + text
        return [TitlePart(content=title, metadata=metadata, level=level)], pos

    def _convert_inline(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        token = tokens[idx]
        img_parts = self._extract_image_parts(ctx, tokens[idx : idx + 1], metadata)
        metadata["md_source_map"] = token.map
        metadata["md_nesting"] = ctx.nesting
        return [TextPart(content=token.content, metadata=metadata)] + img_parts, idx + 1

    def _convert_hr(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        token = tokens[idx]
        metadata["md_source_map"] = token.map
        metadata["md_nesting"] = ctx.nesting
        return [TextPart(content=token.markup, metadata=metadata)], idx + 1

    def _convert_html_block(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        token = tokens[idx]
        metadata["md_source_map"] = token.map
        metadata["md_nesting"] = ctx.nesting
        return [TextPart(content=token.content, metadata=metadata)], idx + 1

    def _convert_paragraph_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        parts, pos = self.convert_until_close(ctx, "paragraph_close", tokens, idx + 1, metadata)
        return parts, pos

    def _convert_ordered_list_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        with self.OrderedListNester(ctx):
            parts, pos = self.convert_until_close(ctx, "ordered_list_close", tokens, idx + 1, metadata)
            return parts, pos

    def _convert_bullet_list_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        with self.Nester(ctx):
            parts, pos = self.convert_until_close(ctx, "bullet_list_close", tokens, idx + 1, metadata)
            return parts, pos

    def _convert_list_item_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        parts, pos = self.convert_until_close(ctx, "list_item_close", tokens, idx + 1, metadata)
        token = tokens[idx]
        if len(token.info) != 0:
            # Item of ordered list
            item_marker = str(ctx.ordinal) + token.markup + " "
            ctx.ordinal += 1
        else:
            # Item of unordered list
            item_marker = token.markup + " "
        if len(parts) == 0:
            # Empty item, e.g. "2. "
            metadata["md_source_map"] = token.map
            metadata["md_nesting"] = ctx.nesting
            return [TextPart(content=item_marker, metadata=metadata)], pos

        result = []
        first_part = parts[0]
        if isinstance(first_part, TextPart):
            # If the first block is a paragraph-like block, then prepend the marker:
            #   item content,     =>  1. item content,
            #   the second line          the second line
            lines = (first_part.content or "").split("\n")
            if len(lines) > 0:
                spaces = " " * len(item_marker)
                lines[0] = item_marker + lines[0]
                for i in range(1, len(lines)):
                    lines[i] = spaces + lines[i]
            else:
                lines.append(item_marker)

            first_part.content = "\n".join(lines)
            result.append(first_part)
        else:
            # If the first block is a code block, or something else,
            # we don't modify the content of the block.
            meta = metadata.copy()
            meta["md_source_map"] = token.map
            meta["md_nesting"] = ctx.nesting
            result.append(TextPart(content=item_marker, metadata=meta))
            result.append(first_part)

        spaces = "    "
        for part in parts[1:]:
            # Adjust indention for paragraph blocks
            if isinstance(part, TextPart):
                lines = (part.content or "").split("\n")
                lines = [spaces + line for line in lines]
                if len(lines) > 0:
                    part.content = "\n".join(lines)

            result.append(part)

        return result, pos

    def _convert_definition(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        token = tokens[idx]
        tm = token.meta or {}
        content = f"[{tm.get('label')}]: {tm.get('url')}"
        title = tm.get("title")
        if title:
            content = content + f" ({title})"
        metadata["md_source_map"] = token.map
        metadata["md_nesting"] = ctx.nesting
        return [TextPart(content=content, metadata=metadata)], idx + 1

    def _convert_table_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        # Image parts can interfere with table processing. For example, a tr might
        # mistakenly identify an ImagePart as a separate column, leading to errors.
        # Therefore, we temporarily disable image extraction and handle it after
        # the entire table has been processed.
        with self.Nester(ctx):
            with self.PauseExtraction(ctx):
                parts, pos = self.convert_until_close(ctx, "table_close", tokens, idx + 1, metadata)
        img_parts = self._extract_image_parts(ctx, tokens[idx:pos], metadata)
        # Parts should contain two items, thead and tbody
        text = "\n".join([part.content for part in parts if part.content is not None])
        metadata["md_source_map"] = tokens[idx].map
        metadata["md_nesting"] = ctx.nesting
        return [TextPart(content=text, metadata=metadata)] + img_parts, pos

    def _convert_thead_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        parts, pos = self.convert_until_close(ctx, "thead_close", tokens, idx + 1, metadata)
        if len(parts) == 0:
            return [], pos
        # Parts should contain one item, which is a tr
        column_count = parts[0].metadata.get("column_count", 0)
        if column_count == 0:
            return [], pos
        text = parts[0].content or ""
        text += "\n" + ("|---" * column_count) + "|"
        return [TextPart(content=text, metadata=metadata)], pos

    def _convert_tbody_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        parts, pos = self.convert_until_close(ctx, "tbody_close", tokens, idx + 1, metadata)
        text = "\n".join([part.content for part in parts if part.content is not None])
        return [TextPart(content=text, metadata=metadata)], pos

    @staticmethod
    def _escape_cell(text: str) -> str:
        text = text.replace("|", "\\|")
        text = text.replace("\r", "")
        text = text.replace("\n", "<br>")
        return text

    def _convert_tr_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        parts, pos = self.convert_until_close(ctx, "tr_close", tokens, idx + 1, metadata)
        if len(parts) == 0:
            return [], pos
        text = ""
        for part in parts:
            text += "| "
            if part.content is not None:
                text += self._escape_cell(part.content) + " "
        text += "|"
        metadata["column_count"] = len(parts)
        return [TextPart(content=text, metadata=metadata)], pos

    def _convert_th_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        parts, pos = self.convert_until_close(ctx, "th_close", tokens, idx + 1, metadata)
        text = ""
        for part in parts:
            if part.content is not None:
                text += part.content
        return [TextPart(content=text, metadata=metadata)], pos

    def _convert_td_open(
        self, ctx: Context, tokens: list[Token], idx: int, metadata: dict[str, Any]
    ) -> tuple[list[Part], int]:
        parts, pos = self.convert_until_close(ctx, "td_close", tokens, idx + 1, metadata)
        text = ""
        for part in parts:
            if part.content is not None:
                text += part.content
        return [TextPart(content=text, metadata=metadata)], pos
