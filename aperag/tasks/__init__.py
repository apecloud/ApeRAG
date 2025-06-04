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

# Import all tasks to register them with celery
from aperag.tasks.collection import delete_collection_task, init_collection_task  # noqa: F401

# You can add other task imports here as needed
# from aperag.tasks.index import *  # noqa: F401
# from aperag.tasks.crawl_web import *  # noqa: F401
# from aperag.tasks.local_directory_task import *  # noqa: F401

__all__ = [
    "init_collection_task",
    "delete_collection_task",
]
