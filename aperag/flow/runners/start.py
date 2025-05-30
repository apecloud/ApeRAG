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

from typing import Tuple

from pydantic import BaseModel, Field

from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner


class StartInput(BaseModel):
    query: str = Field(..., description="User's question or query")


class StartOutput(BaseModel):
    query: str


@register_node_runner(
    "start",
    input_model=StartInput,
    output_model=StartOutput,
)
class StartNodeRunner(BaseNodeRunner):
    async def run(self, ui: StartInput, si: SystemInput) -> Tuple[StartOutput, dict]:
        """
        Run start node. ui: user input; si: system input (SystemInput).
        Returns (output, system_output)
        """
        return StartOutput(query=si.query), {}
