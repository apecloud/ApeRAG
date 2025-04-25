import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, List, Type, TypeVar, Union, get_args

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.data.read_api import read_local_office
from magic_pdf.dict2md.ocr_mkcontent import get_title_level, merge_para_with_text
from magic_pdf.libs import config_reader
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.operators.pipes import PipeResult

from aperag.readers.markdown_reader import MarkdownReader

logger = logging.getLogger(__name__)


OFFICE_DOC_SUFFIXES = set((".doc", ".docx", ".ppt", ".pptx"))
SUPPORTED_SUFFIXES = set((".pdf",)).union(OFFICE_DOC_SUFFIXES)


FallbackReader = TypeVar("FallbackReader")


class MinerUReader(BaseReader, Generic[FallbackReader]):
    _use_fallback_reader: bool = False

    def _set_config_path(self) -> bool:
        path = Path(os.environ.get("MINERU_CONFIG_JSON", "./magic-pdf.json"))
        if not path.exists():
            logger.warning(
                f"MinerUReader config {path} is not found, fallback to use {self._fallback_reader_cls().__name__}."
            )
            return False

        # Adjust the model dir when running in docker.
        new_cache_dir = os.environ.get("MINERU_ADJUST_MODEL_CACHE_DIR", None)
        if new_cache_dir is not None:
            cfg = {}
            with open(str(path), "r") as f:
                cfg = json.loads(f.read())
            self._adjust_cache_path(cfg, "models-dir", new_cache_dir)
            self._adjust_cache_path(cfg, "layoutreader-model-dir", new_cache_dir)
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

    def _adjust_cache_path(self, cfg: dict, key: str, new_cache_dir: str):
        cache_dir_name = ".cache"
        orig = cfg.get(key, "")
        pos = orig.find(cache_dir_name)
        if pos == -1:
            return
        cfg[key] = str(Path(new_cache_dir + "/" + orig[pos + len(cache_dir_name) :]))

    def _fallback_reader_cls(self) -> Type[FallbackReader]:
        return get_args(self.__orig_class__)[0]

    def _check_soffice(self) -> bool:
        if shutil.which("soffice") is None:
            logger.warning(f"soffice command was not found, fallback to use {self._fallback_reader_cls().__name__}.")
            return False
        return True

    def use_fallback_reader(self, v: bool):
        self._use_fallback_reader = v

    def _should_use_fallback_reader(self, file_suffix: str) -> bool:
        if self._use_fallback_reader:
            return True
        if not self._set_config_path():
            return True
        if file_suffix in OFFICE_DOC_SUFFIXES:
            return not self._check_soffice()
        return False

    def load_data(self, file: Path, metadata: dict | None = None) -> list[Document]:
        docs, _ = self.load_data_ex(file, metadata)
        return docs

    def load_data_ex(
        self,
        file: Path,
        metadata: dict | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        tokenizer: Callable[[str], List[int]] | None = None,
    ) -> tuple[list[Document], bool]:
        """Parse file."""

        file_suffix = file.suffix.lower()
        if file_suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type: {file_suffix}. Supported types are: {', '.join(SUPPORTED_SUFFIXES)}"
            )

        if self._should_use_fallback_reader(file_suffix):
            reader_cls = self._fallback_reader_cls()
            docs = reader_cls().load_data(file, metadata)
            return docs, False

        temp_dir = os.environ.get("MINERU_TEMP_FILE_DIR", None)
        temp_dir_obj: tempfile.TemporaryDirectory | None = None
        if not temp_dir:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

        try:
            local_md_dir = os.path.join(temp_dir, "output")
            local_image_dir = os.path.join(temp_dir, "output/images")
            image_dir = str(os.path.basename(local_image_dir))
            input_file_name = file.stem

            os.makedirs(local_image_dir, exist_ok=True)

            image_writer = FileBasedDataWriter(local_image_dir)
            md_writer = FileBasedDataWriter(local_md_dir)

            parse_method = SupportedPdfParseMethod.OCR
            ds: PymuDocDataset = None
            if file.suffix == ".pdf":
                reader1 = FileBasedDataReader("")
                pdf_bytes = reader1.read(str(file))
                ds = PymuDocDataset(pdf_bytes)
                parse_method = ds.classify()
            else:
                # Note: this requires the "soffice" command to convert office docs into PDF.
                # The "soffice" command is part of LibreOffice, can be installed via:
                #   apt-get install libreoffice
                #   brew install libreoffice
                ds = read_local_office(str(file))[0]

            pipe_result: PipeResult = None
            if parse_method == SupportedPdfParseMethod.OCR:
                result = ds.apply(doc_analyze, ocr=True)
                pipe_result = result.pipe_ocr_mode(image_writer)
            else:
                result = ds.apply(doc_analyze, ocr=False)
                pipe_result = result.pipe_txt_mode(image_writer)

            # TODO: save images to s3

            chunked = False
            chunking_params = [chunk_size, chunk_overlap, tokenizer]
            if all(param is not None for param in chunking_params):
                logger.info("Chunking the document using MinerUMiddleJsonChunker.")
                middle_json = pipe_result.get_middle_json()
                docs = MinerUMiddleJsonChunker(chunk_size, chunk_overlap, tokenizer).split(middle_json)
                chunked = True
                return docs, chunked

            # Fallback to the markdown reader
            pipe_result.dump_md(md_writer, f"{input_file_name}.md", image_dir)
            md_file = os.path.join(local_md_dir, f"{input_file_name}.md")
            docs = MarkdownReader().load_data(Path(md_file), metadata=metadata)
            return docs, chunked
        except:
            logger.exception("MinerUReader failed")
            raise
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()


class Type(Enum):
    TEXT = "text"
    TITLE = "title"
    EQUATION = "equation"
    IMAGE = "image"
    TABLE = "table"


class TableFormat(Enum):
    LATEX = "latex"
    HTML = "html"


@dataclass
class Metadata:
    page_idx: int
    bbox: tuple[float, float, float, float]
    titles: list[str] | None


@dataclass
class Elem:
    type: Type
    text: str
    text_level: int  # Always 0 for non-title elem
    metadata: Metadata
    img_path: str | None = None  # Local path of the image, for IMAGE or TABLE
    table_format: TableFormat | None = None
    tokens: int | None = None


@dataclass
class Group:
    elems: List[Union["Group", "Elem"]]
    tokens: int | None = None


class MinerUMiddleJsonChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int, tokenizer: Callable[[str], List[int]]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    def split(self, middle_json: str) -> list[Document]:
        group: Group = middle_json_to_group(middle_json)
        return self._to_docs(group, None, None)

    def _to_docs(self, elem: Group | Elem, prev: Document | None, group_tokens: int | None = None) -> list[Document]:
        if isinstance(elem, Elem):
            tokens = self._count_tokens(elem)

            # Check if the current element should be merged into the previous document.
            merge_to_prev = False
            if prev is not None:
                if elem.type != Type.TITLE:
                    # If the current element is not a TITLE, it can be merged
                    # if the combined token count does not exceed the chunk size.
                    merge_to_prev = tokens + self._get_token_count(prev) <= self.chunk_size
                else:
                    # If the current element is a TITLE, it can be merged only if the combined token count
                    # of the previous document and the entire group does not exceed the chunk size.
                    merge_to_prev = group_tokens + self._get_token_count(prev) <= self.chunk_size
            if merge_to_prev:
                self._append_elem_to_doc(prev, elem)
                return []

            # If the current element's token count is within the chunk size,
            # create a new document with the element's text and metadata.
            if tokens <= self.chunk_size:
                doc = Document(text=elem.text, metadata=self._to_doc_metadata(elem.metadata))
                self._update_token_count(doc, tokens)
                return [doc]

            # Otherwise, split the element's text into smaller chunks and
            # create a document for each chunk.
            docs = []
            chunks = SimpleSemanticSplitter(self.tokenizer).split(elem.text, self.chunk_size, self.chunk_overlap)
            for chunk in chunks:
                doc = Document(text=chunk, metadata=self._to_doc_metadata(elem.metadata))
                self._update_token_count(doc, len(self.tokenizer(chunk)))
                docs.append(doc)
            return docs

        if isinstance(elem, Group):
            group_tokens = self._count_tokens(elem)
            docs = []
            for child in elem.elems:
                docs.extend(self._to_docs(child, prev, group_tokens))
                if len(docs) > 0:
                    prev = docs[-1]
            return docs

        return []

    def _append_elem_to_doc(self, doc: Document, elem: Elem):
        doc.set_content(doc.text + elem.text)
        total_tokens = self._get_token_count(doc) + self._count_tokens(elem)
        self._update_token_count(doc, total_tokens)
        locations = doc.metadata.get("source_locations", [])
        locations.append(
            {
                "page_idx": elem.metadata.page_idx,
                "bbox": elem.metadata.bbox,
            }
        )
        doc.metadata["source_locations"] = locations

    def _to_doc_metadata(self, md: Metadata) -> dict[str, Any]:
        titles = md.titles or []
        return {
            "source_locations": [
                {
                    "page_idx": md.page_idx,
                    "bbox": md.bbox,
                }
            ],
            "titles": titles,
        }

    def _update_token_count(self, doc: Document, tokens: int):
        doc.metadata["tokens"] = tokens

    def _get_token_count(self, doc: Document) -> int:
        return doc.metadata.get("tokens", 0)

    def _count_tokens(self, elem: Group | Elem) -> int:
        if elem.tokens is not None:
            return elem.tokens
        if isinstance(elem, Elem):
            elem.tokens = len(self.tokenizer(elem.text))
            return elem.tokens
        elif isinstance(elem, Group):
            total = 0
            for child in elem.elems:
                num = self._count_tokens(child)
                total += num
            elem.tokens = total
            return elem.tokens
        return 0


def middle_json_to_group(middle_json: str) -> Group:
    curr_group: Group = Group(elems=[])
    group_stack: list[Group] = []
    title_elem_stack: list[Elem] = []
    titles: list[str] = []

    middle: dict[str, Any] = json.loads(middle_json)
    for page_info in middle.get("pdf_info", []):
        paras_of_layout: list[dict[str, Any]] = page_info.get("para_blocks")
        page_idx: int = page_info.get("page_idx")
        if not paras_of_layout:
            continue
        for para_block in paras_of_layout:
            obj = convert_para(para_block, page_idx, titles)
            if obj is None:
                continue

            if not isinstance(obj, Elem) or obj.type != Type.TITLE:
                curr_group.elems.append(obj)
                continue

            curr_title_elem = obj
            curr_title_level = curr_title_elem.text_level

            while len(title_elem_stack) > 0 and title_elem_stack[-1].text_level >= curr_title_level:
                title_elem_stack.pop()
                curr_group = group_stack.pop()

            group_stack.append(curr_group)
            new_group = Group(elems=[curr_title_elem])
            curr_group.elems.append(new_group)
            curr_group = new_group

            title_elem_stack.append(curr_title_elem)
            titles = [elem.text.strip() for elem in title_elem_stack if elem is not None]

            # The metadata for `curr_title_elem` (the current title element) is initially incorrect
            # because it was created using the `titles` from the parent scope during parsing.
            # We need to update its metadata so that the `titles` reflects the correct
            # hierarchy based on the current stack.
            curr_title_elem.metadata.titles = titles

    if len(group_stack) > 0:
        return group_stack[0]
    return curr_group


def convert_para(
    para_block: dict[str, Any],
    page_idx: int,
    titles: list[str] | None = None,
) -> Group | Elem | None:
    para_type = para_block["type"]
    bbox = para_block.get("bbox", (0, 0, 0, 0))
    metadata = Metadata(
        page_idx=page_idx,
        bbox=tuple(bbox),
        titles=titles,
    )

    if para_type in [BlockType.Text, BlockType.List, BlockType.Index]:
        return Elem(
            type=Type.TEXT,
            text=f"{merge_para_with_text(para_block)}\n\n",
            text_level=0,
            metadata=metadata,
        )
    elif para_type == BlockType.Title:
        return Elem(
            type=Type.TITLE,
            text=f"{merge_para_with_text(para_block)}\n\n",
            text_level=get_title_level(para_block),
            metadata=metadata,  # Note: the titles field is not correct and will be fixed later
        )
    elif para_type == BlockType.InterlineEquation:
        return Elem(
            type=Type.EQUATION,
            text=f"{merge_para_with_text(para_block)}\n\n",
            text_level=0,
            metadata=metadata,
        )
    elif para_type == BlockType.Image:
        return _convert_image_para(para_block, metadata)
    elif para_type == BlockType.Table:
        return _convert_table_para(para_block, metadata)

    return None


def _convert_image_para(para_block: dict[str, Any], metadata: Metadata) -> Group | None:
    img_path = None
    text = ""
    for block in para_block["blocks"]:
        if block["type"] == BlockType.ImageBody:
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["type"] == ContentType.Image:
                        if span.get("image_path", ""):
                            img_path = span["image_path"]
                            text = f"[Image: {img_path}]\n"
        if block["type"] == BlockType.ImageCaption:
            text += f"[ImageCaption: {merge_para_with_text(block)}]\n"
        if block["type"] == BlockType.ImageFootnote:
            text += f"[ImageFootnote: {merge_para_with_text(block)}]\n"

    if len(text) == 0:
        return None

    img_elem = Elem(
        type=Type.IMAGE,
        text=f"{text}\n",
        text_level=0,
        metadata=metadata,
        img_path=img_path,
    )
    return img_elem


def _convert_table_para(para_block: dict[str, Any], metadata: Metadata) -> Group | None:
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
                            table_format = TableFormat.LATEX
                        elif span.get("html", ""):
                            table_body = f"\n\n{span['html']}\n\n"
                            table_format = TableFormat.HTML

                        if span.get("image_path", ""):
                            img_path = span["image_path"]

                        if len(table_body) > 0:
                            text = f"Table ({table_format}):\n{table_body}\n"

        if block["type"] == BlockType.TableCaption:
            text += f"[TableCaption: {merge_para_with_text(block)}]\n"
        if block["type"] == BlockType.TableFootnote:
            text += f"[TableFootnote: {merge_para_with_text(block)}]\n"

    if len(text) == 0:
        return None

    table_elem = Elem(
        type=Type.TABLE,
        text=f"{text}\n",
        text_level=0,
        metadata=metadata,
        img_path=img_path,
        table_format=table_format,
    )
    return table_elem


class SimpleSemanticSplitter:
    # List of separators used for splitting text into smaller chunks while preserving semantic coherence.
    # The separators are ordered hierarchically based on their impact on coherence.
    # Separators with less impact (e.g., paragraph breaks) are prioritized (appear earlier).
    # Separators with more impact (e.g., spaces) are used as a last resort (appear later).
    LEVELED_SEPARATORS = [
        ["\n\n"],
        ["\n"],
        ["。”", "！”", "？”"],
        [".\"", "!\"", "?\""],
        ["。", "！", "？"],
        [".", "!", "?"],
        ["；", "，", "、"],
        [";", ","],
        ["》", "）", "】", "」", "’", "”"],
        ["“", ">", ")", "]", "}", "'", '"'],
        [" ", "\t"],
    ]

    def __init__(self, tokenizer: Callable[[str], List[int]]):
        self.tokenizer = tokenizer

    def split(self, s: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        return self._recursive_split(s, chunk_size, chunk_overlap, 0)

    def _fit(self, s: str, chunk_size: int) -> bool:
        return len(self.tokenizer(s)) <= chunk_size

    def _recursive_split(self, s: str, chunk_size: int, chunk_overlap: int, level: int) -> list[str]:
        if len(s) == 0:
            return []
        if len(s) <= 1 or self._fit(s, chunk_size):
            return [s]

        # No more separators can guide semantic segmentation, so split arbitrarily.
        if level >= len(self.LEVELED_SEPARATORS):
            p = len(s) // 2
            left = self._recursive_split(s[:p], chunk_size, chunk_overlap, level + 1)
            overlap = ""
            if chunk_overlap > 0:
                # Extract a substring with size `chunk_overlap` from the right side of the left part (`s[:p]`)
                # to serve as `overlap`.
                # However, `overlap` cannot be equal to `s[:p]`, otherwise the algorithm won't converge.
                # Therefore, use the right half of `s[:p]` for splitting to ensure `overlap` is not equal to `s[:p]`.
                mid = p // 2
                if mid > 0:
                    overlap = self._cut_right_side(s[:p][mid:], chunk_overlap)
            right = self._recursive_split(overlap + s[p:], chunk_size, chunk_overlap, level + 1)
            return left + right

        chunks = [s]
        for sep in self.LEVELED_SEPARATORS[level]:
            new_chunks = []
            for chunk in chunks:
                parts = chunk.split(sep)
                new_chunks.extend([part + sep for part in parts[:-1]])
                new_chunks.append(parts[-1])
            chunks = new_chunks

        new_chunks = []
        for chunk in chunks:
            # If a chunk `chunk` is larger than `chunk_size`, it will be further split into smaller pieces;
            # otherwise, it remains unchanged.
            parts = self._recursive_split(chunk, chunk_size, chunk_overlap, level + 1)
            new_chunks.extend(parts)
        chunks = new_chunks

        # Merge small pieces into larger chunks, ensuring they fit within `chunk_size`.
        chunks = self._merge_small_chunks(chunks, chunk_size)

        return chunks

    def _cut_right_side(self, s: str, chunk_size: int) -> str:
        if len(s) == 0 or self._fit(s, chunk_size):
            return s
        if len(s) <= 1:
            return ""
        left = 0
        right = len(s)
        while left < right:
            mid = (left + right) // 2
            if self._fit(s[mid:], chunk_size):
                right = mid
            else:
                left = mid + 1
        return s[left:]

    def _merge_small_chunks(self, chunks: list[str], chunk_size: int) -> list[str]:
        merged_chunks = []
        current_chunk = ""
        for chunk in chunks:
            if len(current_chunk) == 0:
                current_chunk = chunk
                continue
            if self._fit(current_chunk + chunk, chunk_size):
                current_chunk += chunk
            else:
                merged_chunks.append(current_chunk)
                current_chunk = chunk
        if len(current_chunk) > 0:
            merged_chunks.append(current_chunk)
        return merged_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--middle_json_path", type=str, help="Path to the middle.json file")
    parser.add_argument("--chunk_size", type=int, default=400, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="Chunk overlap")
    parser.add_argument("--encoding_name", type=str, default="cl100k_base", help="Encoding name for tiktoken")
    args = parser.parse_args()
    print(args)

    middle_json = ""
    with open(args.middle_json_path, "r") as f:
        middle_json = f.read()

    import tiktoken

    encoding = tiktoken.get_encoding(args.encoding_name)
    tokenizer = encoding.encode

    chunker = MinerUMiddleJsonChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        tokenizer=tokenizer,
    )
    docs = chunker.split(middle_json)

    for i, doc in enumerate(docs):
        print(f"--- Document {i + 1} ---")
        # Print all metadata on one line
        metadata_str = f"Metadata: tokens={doc.metadata.get('tokens')}, source_locations={doc.metadata.get('source_locations')}, titles={doc.metadata.get('titles')}"
        print(metadata_str)
        # Print the text content on the next line
        print(f"Text:\n{doc.text}\n")
