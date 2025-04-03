
"""Markdown parser.

Contains parser for md files.

"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from llama_index.readers.base import BaseReader
from llama_index.schema import Document
from mdsplit import Chapter, split_by_heading
from pydantic import BaseModel

CHUNK_SPLIT_THRESHOLD = 500


class Group(BaseModel):
    size: int
    chapters: List[Chapter]


class MarkdownReader(BaseReader):
    """Markdown parser.

    Extract text from markdown files.
    Returns dictionary with keys as headers and values as the text between headers.

    """

    def __init__(
        self,
        *args: Any,
        remove_hyperlinks: bool = True,
        remove_images: bool = True,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images

    def markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[List[str]], str]]:
        """Convert a markdown file to a dictionary.

        The keys are the headers and the values are the text under each header.

        """
        markdown_tups: List[Tuple[Optional[List[str]], str]] = []
        lines = markdown_text.split("\n")

        headers = []
        current_level = 0
        current_header = None
        current_text = ""

        def flat_headers():
            result = []
            for h in headers:
                result.append(h[0])
            return result

        for line in lines:
            header_match = re.findall(r"^(#+)\s(.+)$", line)
            if header_match:
                level = len(header_match[0][0])
                if level == current_level:
                    headers[-1] = (headers[-1][0] + line, level)
                elif level > current_level:
                    headers.append((line, level))
                else:
                    for header in reversed(headers):
                        if header[1] > level:
                            headers.pop()
                        else:
                            break
                    if len(headers) > 0:
                        headers[-1] = (line, level)
                    else:
                        headers.append((line, level))

                if len(current_text) < CHUNK_SPLIT_THRESHOLD:
                    # line = line.strip().lstrip("#")
                    current_text += line + "\n"
                else:
                    markdown_tups.append((flat_headers(), current_text))
                    current_text = ""
                current_header = line
                current_level = level
            else:
                current_text += line + "\n"
        if len(current_text) >= CHUNK_SPLIT_THRESHOLD:
            markdown_tups.append((flat_headers(), current_text))
        else:
            if len(markdown_tups) > 0:
                headers, text = markdown_tups[-1]
                text += current_text
                markdown_tups[-1] = (headers, text)
            else:
                markdown_tups.append((flat_headers(), current_text))

        if current_header is not None:
            # pass linting, assert keys are defined
            markdown_tups = [
                ([re.sub(r"#", "", cast(str, key)).strip() for key in headers if key], re.sub(r"<.*?>", "", value))
                for headers, value in markdown_tups
            ]
        else:
            markdown_tups = [
                ([key for key in headers if key], re.sub("\n", "", value)) for headers, value in markdown_tups
            ]

        return markdown_tups

    def remove_images(self, content: str) -> str:
        """Get a dictionary of a markdown file from its path."""
        pattern = r"!{1}\[\[(.*)\]\]"
        content = re.sub(pattern, "", content)
        return content

    def remove_hyperlinks(self, content: str) -> str:
        """Get a dictionary of a markdown file from its path."""
        pattern = r"\[(.*?)\]\((.*?)\)"
        content = re.sub(pattern, r"\1", content)
        return content

    def _init_parser(self) -> Dict:
        """Initialize the parser with the config."""
        return {}

    def parse_tups(
        self, filepath: Path, errors: str = "ignore"
    ) -> List[Tuple[Optional[str], str]]:
        """Parse file into tuples."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)
        if self._remove_images:
            content = self.remove_images(content)
        markdown_tups = self.markdown_to_tups(content)
        return markdown_tups

    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        """Parse file into string."""
        # if file is too small, just return the whole thing
        # tups = self.parse_tups(file)

        tups = []
        header_text = ""
        all_text = ""
        with open(file, "r", encoding="utf-8") as f:
            chapters = split_by_heading(f, 6)
            tup = ([], 0, "")
            for chapter in chapters:
                all_text += "".join(chapter.text)
                if not chapter.heading:
                    tups.append(([], "".join(chapter.text)))
                    continue

                header_text += chapter.heading.full_line

                # if not group.chapters:
                #     group.chapters.append(chapter)
                #     group.size += len(chapter.text)
                # else:
                #     if group.size >= CHUNK_SPLIT_THRESHOLD:
                #         level = group.chapters[0].heading.heading_level
                #         headers = group.chapters[0].parent_headings


                # if tup[1] < chapter.heading.heading_level:
                #     if len(tup[2]) >= CHUNK_SPLIT_THRESHOLD:
                #         tups.append((tup[0], tup[2]))
                #         tup = (chapter.parent_headings, chapter.heading.heading_level, "".join(chapter.text))
                #     else:
                #         text = tup[2] + "".join(chapter.text)
                #         tup = (chapter.parent_headings, chapter.heading.heading_level, text)
                # else:
                #     tups.append((tup[0], tup[2]))
                #     tup = (chapter.parent_headings, chapter.heading.heading_level, "".join(chapter.text))

                if len(tup[2]) >= CHUNK_SPLIT_THRESHOLD:
                    tups.append((tup[0], tup[2]))
                    tup = (chapter.parent_headings, chapter.heading.heading_level, "".join(chapter.text))
                else:
                    text = tup[2] + "".join(chapter.text)
                    tup = (tup[0], chapter.heading.heading_level, text)

            if len(tup[2]) < CHUNK_SPLIT_THRESHOLD and len(tups) > 0:
                previous_tup = tups.pop()
                previous_tup = (previous_tup[0], previous_tup[1] + tup[2])
                tups.append(previous_tup)
            else:
                tups.append((tup[0], tup[2]))

        if len(all_text) == 0:
            return []

        if not metadata:
            metadata = {}
        metadata.update(
            {
                "content_ratio": (len(all_text) - len(header_text)) / len(all_text)
            }
        )

        with open(file) as fd:
            content = fd.read()
            if len(content) < CHUNK_SPLIT_THRESHOLD:
                return [Document(text=all_text, metadata=metadata)]

        results = []
        # TODO: don't include headers right now
        for headers, value in tups:
            text = value
            if headers:
                header = " ".join(headers)
                text = f"Title: {header}\n\n{value}"
            results.append(Document(text=text, metadata=metadata))
        return results
