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


def parse_json(resp: str) -> dict:
    """
    Parses JSON data from an LLM response string.
    This function handles cases where the JSON is wrapped in markdown code blocks
    like "```json ... ```" or "``` ... ```".

    Args:
        resp: The LLM response string, potentially containing JSON.
    Returns:
        A dictionary parsed from the JSON string.
    """
    json_str = (resp or "").strip()
    if len(resp) > 6:
        if resp.startswith("```json") and resp.endswith("```"):
            json_str = resp[7:][:-3]
        elif resp.startswith("```") and resp.endswith("```"):
            json_str = resp[3:][:-3]
    return json.loads(json_str)
