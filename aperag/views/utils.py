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

import logging
import uuid
from typing import Tuple

from aperag.schema.view_models import CollectionConfig
from aperag.source.base import CustomSourceInitializationError, get_source
from aperag.utils.utils import AVAILABLE_SOURCE

logger = logging.getLogger(__name__)


def validate_source_connect_config(config: CollectionConfig) -> Tuple[bool, str]:
    if config.source is None:
        return False, ""
    if config.source not in AVAILABLE_SOURCE:
        return False, ""
    try:
        get_source(config)
    except CustomSourceInitializationError as e:
        return False, str(e)
    return True, ""


def validate_url(url):
    from urllib.parse import urlparse

    try:
        parsed_url = urlparse(url)

        if parsed_url.scheme not in ["http", "https"]:
            return False

        if not parsed_url.netloc:
            return False

        return True
    except Exception:
        return False


def mask_api_key(api_key: str) -> str:
    """
    Mask API key for security, showing only first 4 and last 4 characters.
    The number of mask characters reflects the actual length of the hidden part.

    Args:
        api_key: The original API key

    Returns:
        Masked API key string

    Examples:
        - sk-1234567890abcdef -> sk-1********cdef (16 chars total, 8 masked)
        - short_key -> short_key (if length <= 8, return as-is)
        - very_long_api_key_123456789 -> very********************6789 (25 chars total, 17 masked)
    """
    if not api_key or len(api_key) <= 8:
        return api_key

    # Calculate the number of characters to mask (total - first 4 - last 4)
    masked_length = len(api_key) - 8
    mask_chars = "*" * masked_length

    # Show first 4 and last 4 characters, mask the middle with actual length
    return f"{api_key[:4]}{mask_chars}{api_key[-4:]}"


def generate_random_provider_name() -> str:
    """
    Generate a random provider name using UUID.

    Returns:
        A random provider name in the format: provider_xxxxxxxx

    Examples:
        - provider_a1b2c3d4
        - provider_f7e8d9c0
    """
    # Generate a short UUID (first 8 characters)
    short_uuid = str(uuid.uuid4()).replace("-", "")[:8]
    return f"provider_{short_uuid}"


def is_google_oauth_enabled() -> bool:
    """
    Check if Google OAuth is enabled based on configuration.

    Returns:
        bool: True if Google OAuth is enabled, False otherwise
    """
    from aperag.config import settings

    return bool(settings.google_oauth_client_id and settings.google_oauth_client_secret)


def is_github_oauth_enabled() -> bool:
    """
    Check if GitHub OAuth is enabled based on configuration.

    Returns:
        bool: True if GitHub OAuth is enabled, False otherwise
    """
    from aperag.config import settings

    return bool(settings.github_oauth_client_id and settings.github_oauth_client_secret)


def get_available_login_methods() -> list[str]:
    """
    Get list of available login methods based on configuration.

    Returns:
        list[str]: List of available login methods
    """
    methods = ["local"]
    if is_google_oauth_enabled():
        methods.append("google")
    if is_github_oauth_enabled():
        methods.append("github")
    return methods
