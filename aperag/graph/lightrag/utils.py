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

"""
LightRAG Module for ApeRAG

This module is based on the original LightRAG project with extensive modifications.

Original Project:
- Repository: https://github.com/HKUDS/LightRAG
- Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (arXiv:2410.05779)
- Authors: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- License: MIT License

Modifications by ApeRAG Team:
- Removed global state management for true concurrent processing
- Added stateless interfaces for Celery/Prefect integration
- Implemented instance-level locking mechanism
- Enhanced error handling and stability
- See changelog.md for detailed modifications
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from dataclasses import dataclass
from hashlib import md5
from typing import Any, Callable, List, Protocol

import numpy as np


def get_env_value(env_key: str, default: any, value_type: type = str, special_none: bool = False) -> any:
    """
    Get value from environment variable with type conversion

    Args:
        env_key (str): Environment variable key
        default (any): Default value if env variable is not set
        value_type (type): Type to convert the value to
        special_none (bool): If True, return None when value is "None"

    Returns:
        any: Converted value from environment or default
    """
    value = os.getenv(env_key)
    if value is None:
        return default

    # Handle special case for "None" string
    if special_none and value == "None":
        return None

    if value_type is bool:
        return value.lower() in ("true", "1", "yes", "t", "on")
    try:
        return value_type(value)
    except (ValueError, TypeError):
        return default


# # Use TYPE_CHECKING to avoid circular imports
# if TYPE_CHECKING:
#     from aperag.graph.lightrag.base import BaseKVStorage

# Initialize logger - smart configuration that follows system defaults
logger = logging.getLogger("GraphIndex")

# Let it propagate to parent loggers (respects system configuration)
logger.propagate = False

# Only add a simple handler if root logger has no handlers (fallback for console output)
root_logger = logging.getLogger()
if not root_logger.handlers:
    # Add a basic console handler only as fallback
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Use INFO level as reasonable default, but it will be overridden by system config when available
    logger.setLevel(logging.INFO)

# Set httpx logging level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    # concurrent_limit: int = 16

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def compute_mdhash_id(content: str, prefix: str = "", workspace: str = "") -> str:
    """
    Compute a unique ID for a given content string with workspace isolation.

    The ID is a combination of the given prefix and the MD5 hash of the content string
    combined with workspace information to ensure proper data isolation.

    Args:
        content: The content string to hash
        prefix: The prefix to add to the hash
        workspace: The workspace identifier for data isolation

    Returns:
        A unique ID string that includes workspace isolation
    """
    # Combine content with workspace to ensure isolation
    if workspace:
        hash_input = f"{workspace}::{content}"
    else:
        hash_input = content
    return prefix + md5(hash_input.encode()).hexdigest()


class TokenizerInterface(Protocol):
    """
    Defines the interface for a tokenizer, requiring encode and decode methods.
    """

    def encode(self, content: str) -> List[int]:
        """Encodes a string into a list of tokens."""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of tokens into a string."""
        ...


class Tokenizer:
    """
    A wrapper around a tokenizer to provide a consistent interface for encoding and decoding.
    """

    def __init__(self, model_name: str, tokenizer: TokenizerInterface):
        """
        Initializes the Tokenizer with a tokenizer model name and a tokenizer instance.

        Args:
            model_name: The associated model name for the tokenizer.
            tokenizer: An instance of a class implementing the TokenizerInterface.
        """
        self.model_name: str = model_name
        self.tokenizer: TokenizerInterface = tokenizer

    def encode(self, content: str) -> List[int]:
        """
        Encodes a string into a list of tokens using the underlying tokenizer.

        Args:
            content: The string to encode.

        Returns:
            A list of integer tokens.
        """
        return self.tokenizer.encode(content)

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of tokens into a string using the underlying tokenizer.

        Args:
            tokens: A list of integer tokens to decode.

        Returns:
            The decoded string.
        """
        return self.tokenizer.decode(tokens)


class TiktokenTokenizer(Tokenizer):
    """
    A Tokenizer implementation using the tiktoken library.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the TiktokenTokenizer with a specified model name.

        Args:
            model_name: The model name for the tiktoken tokenizer to use.  Defaults to "gpt-4o-mini".

        Raises:
            ImportError: If tiktoken is not installed.
            ValueError: If the model_name is invalid.
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is not installed. Please install it with `pip install tiktoken` or define custom `tokenizer_func`."
            )

        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
            super().__init__(model_name=model_name, tokenizer=tokenizer)
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}.")


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": content} for i, content in enumerate(args)]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(
    list_data: list[Any],
    key: Callable[[Any], str],
    max_token_size: int,
    tokenizer: Tokenizer,
) -> list[int]:
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer.encode(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def process_combine_contexts(*context_lists):
    """
    Combine multiple context lists and remove duplicate content

    Args:
        *context_lists: Any number of context lists

    Returns:
        Combined context list with duplicates removed
    """
    seen_content = {}
    combined_data = []

    # Iterate through all input context lists
    for context_list in context_lists:
        if not context_list:  # Skip empty lists
            continue
        for item in context_list:
            content_dict = {k: v for k, v in item.items() if k != "id"}
            content_key = tuple(sorted(content_dict.items()))
            if content_key not in seen_content:
                seen_content[content_key] = item
                combined_data.append(item)

    # Reassign IDs
    for i, item in enumerate(combined_data):
        item["id"] = str(i + 1)

    return combined_data


def get_conversation_turns(conversation_history: list[dict[str, Any]], num_turns: int) -> str:
    """
    Process conversation history to get the specified number of complete turns.

    Args:
        conversation_history: List of conversation messages in chronological order
        num_turns: Number of complete turns to include

    Returns:
        Formatted string of the conversation history
    """
    # Check if num_turns is valid
    if num_turns <= 0:
        return ""

    # Group messages into turns
    turns: list[list[dict[str, Any]]] = []
    messages: list[dict[str, Any]] = []

    # First, filter out keyword extraction messages
    for msg in conversation_history:
        if msg["role"] == "assistant" and (
            msg["content"].startswith('{ "high_level_keywords"') or msg["content"].startswith("{'high_level_keywords'")
        ):
            continue
        messages.append(msg)

    # Then process messages in chronological order
    i = 0
    while i < len(messages) - 1:
        msg1 = messages[i]
        msg2 = messages[i + 1]

        # Check if we have a user-assistant or assistant-user pair
        if (msg1["role"] == "user" and msg2["role"] == "assistant") or (
            msg1["role"] == "assistant" and msg2["role"] == "user"
        ):
            # Always put user message first in the turn
            if msg1["role"] == "assistant":
                turn = [msg2, msg1]  # user, assistant
            else:
                turn = [msg1, msg2]  # user, assistant
            turns.append(turn)
        i += 2

    # Keep only the most recent num_turns
    if len(turns) > num_turns:
        turns = turns[-num_turns:]

    # Format the turns into a string
    formatted_turns: list[str] = []
    for turn in turns:
        formatted_turns.extend([f"user: {turn[0]['content']}", f"assistant: {turn[1]['content']}"])

    return "\n".join(formatted_turns)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


def lazy_external_import(module_name: str, class_name: str) -> Callable[..., Any]:
    """Lazily import a class from an external module based on the package of the caller."""
    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args: Any, **kwargs: Any):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


def get_content_summary(content: str, max_length: int = 250) -> str:
    """Get summary of document content

    Args:
        content: Original document content
        max_length: Maximum length of summary

    Returns:
        Truncated content with ellipsis if needed
    """
    content = content.strip()
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def normalize_extracted_info(name: str, is_entity=False) -> str:
    """Normalize entity/relation names and description with the following rules:
    1. Remove spaces between Chinese characters
    2. Remove spaces between Chinese characters and English letters/numbers
    3. Preserve spaces within English text and numbers
    4. Replace Chinese parentheses with English parentheses
    5. Replace Chinese dash with English dash
    6. Remove English quotation marks from the beginning and end of the text
    7. Remove English quotation marks in and around chinese
    8. Remove Chinese quotation marks

    Args:
        name: Entity name to normalize

    Returns:
        Normalized entity name
    """
    # Replace Chinese parentheses with English parentheses
    name = name.replace("（", "(").replace("）", ")")

    # Replace Chinese dash with English dash
    name = name.replace("—", "-").replace("－", "-")

    # Use regex to remove spaces between Chinese characters
    # Regex explanation:
    # (?<=[\u4e00-\u9fa5]): Positive lookbehind for Chinese character
    # \s+: One or more whitespace characters
    # (?=[\u4e00-\u9fa5]): Positive lookahead for Chinese character
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)

    # Remove spaces between Chinese and English/numbers/symbols
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name)
    name = re.sub(r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name)

    # Remove English quotation marks from the beginning and end
    if len(name) >= 2 and name.startswith('"') and name.endswith('"'):
        name = name[1:-1]
    if len(name) >= 2 and name.startswith("'") and name.endswith("'"):
        name = name[1:-1]

    if is_entity:
        # remove Chinese quotes
        name = name.replace(""", "").replace(""", "").replace("'", "").replace("'", "")
        # remove English queotes in and around chinese
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)

    return name


def clean_text(text: str) -> str:
    """Clean text by removing null bytes (0x00) and whitespace

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    return text.strip().replace("\x00", "")


class LightRAGLogger:
    """
    Unified logger for LightRAG processing progress.
    Replaces the legacy pipeline_status system with structured logging.
    """

    def __init__(self, prefix: str = "Graph Index", workspace: str = "default"):
        """
        Initialize the logger with custom prefix and workspace.

        Args:
            prefix: Log message prefix (default: "LightRAG")
            workspace: Workspace identifier for multi-tenant logging
        """
        self.prefix = prefix
        self.workspace = workspace

    def _format_message(self, message: str, level: str = "INFO") -> str:
        """Format log message with prefix and workspace."""
        workspace_info = f"[{self.workspace}]"

        return f"{self.prefix}{workspace_info} {message}"

    def info(self, message: str):
        """Log info level message."""
        formatted_msg = self._format_message(message, "INFO")
        logger.info(formatted_msg)

    def warning(self, message: str):
        """Log warning level message."""
        formatted_msg = self._format_message(message, "WARNING")
        logger.warning(formatted_msg)

    def error(self, message: str):
        """Log error level message."""
        formatted_msg = self._format_message(message, "ERROR")
        logger.error(formatted_msg)

    def debug(self, message: str):
        """Log debug level message."""
        formatted_msg = self._format_message(message, "DEBUG")
        logger.debug(formatted_msg)

    def log_extraction_progress(self, current_chunk: int, total_chunks: int, entities_count: int, relations_count: int):
        """Log chunk extraction progress."""
        message = f"Chunk {current_chunk} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel"
        self.debug(message)

    def log_entity_merge(
        self, entity_name: str, total_fragments: int, new_fragments: int, is_llm_summary: bool = False
    ):
        """Log entity merge operations."""
        prefix = "LLM merge N" if is_llm_summary else "Merge N"
        message = f"{prefix}: {entity_name} | {new_fragments}+{total_fragments - new_fragments}"
        self.debug(message)

    def log_relation_merge(
        self, src_id: str, tgt_id: str, total_fragments: int, new_fragments: int, is_llm_summary: bool = False
    ):
        """Log relation merge operations."""
        prefix = "LLM merge E" if is_llm_summary else "Merge E"
        message = f"{prefix}: {src_id} - {tgt_id} | {new_fragments}+{total_fragments - new_fragments}"
        self.debug(message)

    def log_timing(self, operation: str, duration: float, details: str = ""):
        """Log operation timing with emoji and formatted duration."""
        details_str = f" ({details})" if details else ""
        self.info(f"Graph Index function duration:  {operation}: {duration:.3f}s{details_str}")


def timing_wrapper(operation_name: str):
    """Simple timing decorator for async functions that auto-detects lightrag_logger from function arguments"""
    import time

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Try to find lightrag_logger in function arguments
            lightrag_logger = kwargs.get("lightrag_logger")
            if not lightrag_logger and args:
                # Search in args for LightRAGLogger instance
                for arg in args:
                    if isinstance(arg, LightRAGLogger):
                        lightrag_logger = arg
                        break

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                if duration > 5:
                    if lightrag_logger:
                        lightrag_logger.log_timing(operation_name, duration)
                    else:
                        logger.info(f"Graph Index function duration:️  {operation_name}: {duration:.3f}s")
                return result
            except Exception:
                duration = time.time() - start_time
                if lightrag_logger:
                    lightrag_logger.log_timing(f"{operation_name} (FAILED)", duration)
                else:
                    logger.info(f"Graph Index function duration:️  {operation_name} (FAILED): {duration:.3f}s")
                raise

        return wrapper

    return decorator


def create_lightrag_logger(prefix: str = "Graph Index", workspace: str = "default") -> LightRAGLogger:
    """Create a LightRAGLogger instance with specified prefix and workspace."""
    return LightRAGLogger(prefix=prefix, workspace=workspace)
