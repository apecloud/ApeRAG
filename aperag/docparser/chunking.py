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

from dataclasses import dataclass
from typing import Callable, List

from aperag.docparser.base import Part


def rechunk(
    parts: list[Part], chunk_size: int, chunk_overlap: int, tokenizer: Callable[[str], List[int]]
) -> list[Part]:
    rechunker = Rechunker(chunk_size, chunk_overlap, tokenizer)
    return rechunker(parts)


@dataclass
class Group:
    title_level: int
    title: str
    items: list[Part]
    tokens: int | None = None


class Rechunker:
    def __init__(self, chunk_size: int, chunk_overlap: int, tokenizer: Callable[[str], List[int]]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    def __call__(self, parts: list[Part]) -> list[Part]:
        groups = self._to_groups(parts)
        return self._rechunk(groups)

    def _to_groups(self, parts: list[Part]) -> list[Group]:
        result: list[Group] = []
        curr_group: Group | None = None

        for part in parts:
            if not part.content:
                continue

            nesting = part.metadata.get("nesting", 0)
            title_level = 0
            title = ""
            if hasattr(part, "level"):  # TitlePart
                title_level = part.level
                title = part.content or ""

            if curr_group is None:
                curr_group = Group(title_level=title_level, title=title, items=[part])
                result.append(curr_group)
                continue

            # For simplicity, titles within lower-level nesting will not create new groups.
            if title_level == 0 or nesting != 0:
                curr_group.items.append(part)
                continue

            curr_group = Group(title_level=title_level, title=title, items=[part])
            result.append(curr_group)

        return result

    def _rechunk(self, groups: list[Group]) -> list[Part]:
        title_stack: list[tuple[str, int]] = []
        titles: list[str] = []
        result: list[Part] = []
        last_part: Part | None = None
        highest_level_in_last_part: int | None = None

        for group in groups:
            while len(title_stack) > 0 and title_stack[-1][1] >= group.title_level:
                title_stack.pop()
            if group.title_level > 0:
                title_stack.append((group.title, group.title_level))
            titles = [tup[0] for tup in title_stack]

            group_tokens = self._count_tokens(group)

            # Check if the group can be merged into the last Part
            can_merge = True
            if highest_level_in_last_part is not None and highest_level_in_last_part > group.title_level:
                # Do not merge if the current group has a higher title level
                # (e.g., merging content under a main heading into a sub-heading)
                can_merge = False
            last_part_tokens = 0 if last_part is None else self._count_tokens(last_part)
            if last_part_tokens + group_tokens > self.chunk_size:
                can_merge = False

            if can_merge:
                last_part = self._append_group_to_part(group, last_part, titles)
                if highest_level_in_last_part is None:
                    highest_level_in_last_part = group.title_level
                continue

            # Since the current group can't be merged into the last part,
            # the last part can be sealed.
            if last_part is not None:
                result.append(last_part)
                last_part = None
                highest_level_in_last_part = None

            # Split large parts
            parts: list[Part] = []
            for part in group.items:
                tokens = self._count_tokens(part)
                if tokens > self.chunk_size:
                    # If the single part is too large, split it into smaller chunks
                    splitter = SimpleSemanticSplitter(self.tokenizer)
                    chunks = splitter.split(part.content, self.chunk_size, self.chunk_overlap)
                    metadata = part.metadata
                    metadata.pop("tokens", None)
                    for chunk in chunks:
                        parts.append(Part(content=chunk, metadata=metadata.copy()))
                else:
                    parts.append(part)

            # Rechunk the parts
            assert last_part is None
            highest_level_in_last_part = group.title_level  # All parts are in the same group
            tokens_sum = 0
            for part in parts:
                tokens = self._count_tokens(part)
                if tokens_sum + tokens > self.chunk_size:
                    if last_part is not None:
                        result.append(last_part)
                        last_part = None
                        tokens_sum = 0

                last_part = self._append_part_to_part(part, last_part, titles)
                tokens_sum += tokens

        if last_part is not None:
            result.append(last_part)

        return result

    def _append_group_to_part(self, group: Group, dest: Part | None, titles: list[str]) -> Part:
        for part in group.items:
            dest = self._append_part_to_part(part, dest, titles)
        return dest

    def _append_part_to_part(self, part: Part, dest: Part | None, titles: list[str]) -> Part:
        if dest is None:
            metadata = part.metadata.copy()
            if titles and titles[-1] == part.content:
                # Remove the last title if it is exactly the current part
                titles = titles[:-1]
            if titles:
                metadata["titles"] = titles.copy()
            # Normalize to a Part
            return Part(content=part.content, metadata=metadata)
        dest.content += "\n\n" + part.content
        self._merge_md_source_map(dest, part)
        self._merge_pdf_source_map(dest, part)
        dest.metadata.pop("tokens", None)
        return dest

    def _merge_md_source_map(self, dest: Part, src: Part):
        dest_map = dest.metadata.get("md_source_map", None)
        src_map = src.metadata.get("md_source_map", None)
        if dest_map is None and src_map is None:
            return
        if dest_map is not None and src_map is None:
            return
        if dest_map is None and src_map is not None:
            dest.metadata["md_source_map"] = src_map
            return
        new_map = [min(dest_map[0], src_map[0]), max(dest_map[1], src_map[1])]
        dest.metadata["md_source_map"] = new_map

    def _merge_pdf_source_map(self, dest: Part, src: Part):
        dest_map: list[dict] = dest.metadata.get("pdf_source_map", None)
        src_map: list[dict] = src.metadata.get("pdf_source_map", None)
        if dest_map is None and src_map is None:
            return
        if dest_map is not None and src_map is None:
            return
        if dest_map is None and src_map is not None:
            dest.metadata["pdf_source_map"] = src_map
            return
        new_map = dest_map
        for item in src_map:
            if item not in new_map:
                new_map.append(item)
        dest.metadata["pdf_source_map"] = new_map

    def _count_tokens(self, elem: Group | Part) -> int:
        if isinstance(elem, Group):
            if elem.tokens is not None:
                return elem.tokens
            total = 0
            for child in elem.items:
                num = self._count_tokens(child)
                total += num
            elem.tokens = total
            return elem.tokens
        else:
            # elem is a Part
            tokens = elem.metadata.get("tokens", None)
            if tokens is not None:
                return tokens
            tokens = len(self.tokenizer(elem.content))
            elem.metadata["tokens"] = tokens
            return tokens


class SimpleSemanticSplitter:
    # List of separators used for splitting text into smaller chunks while preserving semantic coherence.
    # The separators are ordered hierarchically based on their impact on coherence.
    # Separators with less impact (e.g., paragraph breaks) are prioritized (appear earlier).
    # Separators with more impact (e.g., spaces) are used as a last resort (appear later).
    LEVELED_SEPARATORS = [
        ["\n\n"],
        ["\n"],
        ["。”", "！”", "？”"],
        ['."', '!"', '?"'],
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
