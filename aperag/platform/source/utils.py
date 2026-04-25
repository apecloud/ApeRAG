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

import asyncio
import functools
import os
import tempfile


def gen_temporary_file(name, default_suffix=""):
    prefix, suffix = os.path.splitext(name)
    prefix = prefix.strip("/").replace("/", "--")
    suffix = suffix.lower()
    if not suffix:
        suffix = default_suffix
    return tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=suffix)


async def async_run(f, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(f, *args, **kwargs))


def find_duplicate_paths(paths):
    """
    Find if there are any duplicate paths in terms of parent-child relationships.

    Args:
    paths (list of str): A list of file or directory paths.

    Returns:
    list of tuples: Each tuple contains a pair of paths where one is the parent of the other.
    """
    sorted_paths = sorted(paths, key=lambda path: path.count(os.sep))  # Sort by depth
    duplicate_paths = []

    for i, path in enumerate(sorted_paths):
        for other_path in sorted_paths[i + 1 :]:
            if other_path.startswith(path + os.sep):  # Check for parent-child relationship
                duplicate_paths.append((path, other_path))

    return duplicate_paths
