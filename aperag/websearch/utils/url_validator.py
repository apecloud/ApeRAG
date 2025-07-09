"""
URL Validator

Utility class for URL validation and normalization.
"""

import re
from typing import List
from urllib.parse import urlparse


class URLValidator:
    """
    URL validation and normalization utility.
    """

    # Basic URL regex pattern
    URL_PATTERN = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Check if URL is valid.

        Args:
            url: URL to validate

        Returns:
            True if valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False

        try:
            result = urlparse(url)
            return (
                result.scheme in ("http", "https") and result.netloc and URLValidator.URL_PATTERN.match(url) is not None
            )
        except Exception:
            return False

    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URL (add protocol if missing, etc.).

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        if not url:
            return url

        url = url.strip()

        # Add protocol if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        return url

    @staticmethod
    def extract_domain(url: str) -> str:
        """
        Extract domain from URL.

        Args:
            url: URL to extract domain from

        Returns:
            Domain name
        """
        try:
            result = urlparse(url)
            return result.netloc.lower()
        except Exception:
            return ""

    @staticmethod
    def validate_urls(urls: List[str]) -> List[str]:
        """
        Validate a list of URLs and return only valid ones.

        Args:
            urls: List of URLs to validate

        Returns:
            List of valid URLs
        """
        return [url for url in urls if URLValidator.is_valid_url(url)]
