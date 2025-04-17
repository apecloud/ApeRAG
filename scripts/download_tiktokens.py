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

import hashlib
import os
import urllib
import urllib.request


def download_tiktoken_file(url: str, cache_dir: str):
    # Expand the user directory if '~' is used
    cache_dir = os.path.expanduser(cache_dir)

    # Calculate the SHA1 hash of the URL and get the first 40 chars
    url_bytes = url.encode('utf-8')
    sha1_hash = hashlib.sha1(url_bytes).hexdigest()
    filename = sha1_hash[:40]

    # Construct the full destination path
    destination_path = os.path.join(cache_dir, filename)

    if os.path.exists(destination_path):
        print(f"File {destination_path} already exists, skipping download.")
        return

    # Ensure the cache directory exists
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Ensured cache directory exists: {cache_dir}")
    except OSError as e:
        print(f"Error creating directory {cache_dir}: {e}")
        exit(1)

    # Download the file
    print(f"Downloading '{url}' to '{destination_path}'...")
    try:
        urllib.request.urlretrieve(url, destination_path)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading file: {e}")
        os.remove(destination_path)
        exit(1)


def download_tiktokens():
    default_cache_dir = "~/.cache/tiktoken"
    cache_dir = os.environ.get("CACHE_DIR", default_cache_dir)

    models = [
        "o200k_base",  # for o1-, o3-, gpt-4o-
        "cl100k_base", # for gpt-4-, or others
    ]
    for model in models:
        download_url = f"https://openaipublic.blob.core.windows.net/encodings/{model}.tiktoken"
        download_tiktoken_file(download_url, cache_dir)

    # Since TIKTOKEN_CACHE_DIR doesn't support expanding the user directory (i.e. "~/.cache"),
    # we have to point TIKTOKEN_CACHE_DIR to a relative path ".cache/tiktoken".
    # https://github.com/openai/tiktoken/blob/4560a8896f5fb1d35c6f8fd6eee0399f9a1a27ca/tiktoken/load.py#L35
    #
    # Create a symlink from "~/.cache/tiktoken" to ".cache/tiktoken"
    if cache_dir == default_cache_dir and not os.path.exists(".cache/tiktoken"):
        os.makedirs(".cache", exist_ok=True)
        os.symlink(os.path.expanduser(cache_dir), ".cache/tiktoken", target_is_directory=True)


if __name__ == "__main__":
    download_tiktokens()
