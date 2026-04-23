"""
URL Validator

Utility class for URL validation and normalization.
"""

import logging
import re
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


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

    # Domain name regex pattern
    DOMAIN_PATTERN = re.compile(
        r"^(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}$|"  # domain
        r"^localhost$",  # localhost
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
        if not url or not url.strip():
            return False
        return bool(URLValidator.URL_PATTERN.match(url.strip()))

    @staticmethod
    def is_valid_domain(domain: str) -> bool:
        """
        Check if domain is valid.

        Args:
            domain: Domain to validate

        Returns:
            True if valid, False otherwise
        """
        if not domain or not domain.strip():
            return False
        return bool(URLValidator.DOMAIN_PATTERN.match(domain.strip()))

    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URL by removing trailing slashes and fragments.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        if not url:
            return ""

        url = url.strip()
        # Remove trailing slash
        if url.endswith("/") and len(url) > 1:
            url = url[:-1]

        # Remove fragment
        if "#" in url:
            url = url.split("#")[0]

        return url

    @staticmethod
    def extract_domain(url: str) -> str:
        """
        Extract domain from URL.

        Args:
            url: URL to extract domain from

        Returns:
            Domain name (without port)
        """
        try:
            result = urlparse(url)
            # Remove port from netloc if present
            netloc = result.netloc.lower()
            if ":" in netloc:
                netloc = netloc.split(":")[0]
            return netloc
        except Exception:
            return ""

    @staticmethod
    def extract_domain_from_source(source: str) -> Optional[str]:
        """
        Extract domain from a single source (URL or domain name).

        Args:
            source: Source string (URL or domain name)

        Returns:
            Domain name if valid, None otherwise
        """
        if not source or not source.strip():
            return None

        source = source.strip()

        # Check if it's already a domain name
        if URLValidator.is_valid_domain(source):
            return source.lower()

        # Check if it's a URL
        if URLValidator.is_valid_url(source):
            domain = URLValidator.extract_domain(source)
            if domain:
                return domain

        # If it looks like a domain but failed validation, try to extract anyway
        try:
            parsed = urlparse(source)
            if parsed.netloc:
                # Remove port from netloc if present
                netloc = parsed.netloc.lower()
                if ":" in netloc:
                    netloc = netloc.split(":")[0]
                return netloc
        except Exception as e:
            logger.warning(f"Failed to parse source '{source}': {e}")

        # Try as plain domain
        if "." in source and not source.startswith("http"):
            # Remove protocol if accidentally included
            clean_source = source.replace("http://", "").replace("https://", "")
            if URLValidator.is_valid_domain(clean_source):
                return clean_source.lower()

        logger.warning(f"Invalid source format: '{source}'")
        return None

    @staticmethod
    def extract_domains_from_sources(sources: List[str]) -> List[str]:
        """
        Extract valid domains from a list of sources (URLs or domain names).

        Args:
            sources: List of source strings (URLs or domain names)

        Returns:
            Sorted list of unique domain names
        """
        if not sources:
            return []

        domains = set()

        for source in sources:
            domain = URLValidator.extract_domain_from_source(source)
            if domain:
                domains.add(domain)

        # Return sorted list for consistent output
        return sorted(list(domains))

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
