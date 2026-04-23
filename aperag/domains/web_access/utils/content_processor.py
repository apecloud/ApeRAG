"""
Content Processor

Utility class for processing web content.
"""

import re
from typing import Optional


class ContentProcessor:
    """
    Utility class for processing and cleaning web content.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def extract_title_from_content(content: str) -> Optional[str]:
        """
        Extract title from markdown content.

        Args:
            content: Markdown content

        Returns:
            Extracted title or None
        """
        if not content:
            return None

        # Look for first H1 heading
        h1_match = re.search(r"^# (.+)$", content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()

        # Look for first H2 heading if no H1
        h2_match = re.search(r"^## (.+)$", content, re.MULTILINE)
        if h2_match:
            return h2_match.group(1).strip()

        return None

    @staticmethod
    def count_words(text: str) -> int:
        """
        Count words in text.

        Args:
            text: Text to count words in

        Returns:
            Word count
        """
        if not text:
            return 0

        # Simple word counting - can be enhanced for different languages
        words = re.findall(r"\b\w+\b", text)
        return len(words)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Rough estimation: ~0.75 tokens per word for English text
        # For Chinese text, roughly 1.5 characters per token
        word_count = ContentProcessor.count_words(text)

        # Count Chinese characters
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))

        # Estimate tokens
        english_tokens = word_count * 0.75
        chinese_tokens = chinese_chars / 1.5

        return int(english_tokens + chinese_tokens)

    @staticmethod
    def truncate_content(content: str, max_length: int = 10000) -> str:
        """
        Truncate content to maximum length.

        Args:
            content: Content to truncate
            max_length: Maximum length

        Returns:
            Truncated content
        """
        if not content or len(content) <= max_length:
            return content

        # Try to truncate at word boundary
        truncated = content[:max_length]
        last_space = truncated.rfind(" ")

        if last_space > max_length * 0.8:  # If we find a space reasonably close
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."

    @staticmethod
    def sanitize_markdown(content: str) -> str:
        """
        Sanitize markdown content (remove potentially harmful elements).

        Args:
            content: Markdown content

        Returns:
            Sanitized content
        """
        if not content:
            return ""

        # Remove script tags and their content
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove style tags and their content
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove JavaScript event handlers
        content = re.sub(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', "", content, flags=re.IGNORECASE)

        return content
