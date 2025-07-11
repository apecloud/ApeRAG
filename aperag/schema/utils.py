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

import json

from aperag.schema.view_models import CollectionConfig


def parseCollectionConfig(config: str) -> CollectionConfig:
    try:
        config_dict = json.loads(config)
        collection_config = CollectionConfig.model_validate(config_dict)
        return collection_config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to parse collection config: {str(e)}")


def dumpCollectionConfig(collection_config: CollectionConfig) -> str:
    return collection_config.model_dump_json()
