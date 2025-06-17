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


from aperag.exceptions import invalid_param
from aperag.llm.prompts import MULTI_ROLE_EN_PROMPT_TEMPLATES, MULTI_ROLE_ZH_PROMPT_TEMPLATES
from aperag.schema import view_models


def list_prompt_templates(language: str) -> view_models.PromptTemplateList:
    if language == "zh-CN":
        templates = MULTI_ROLE_ZH_PROMPT_TEMPLATES
    elif language == "en-US":
        templates = MULTI_ROLE_EN_PROMPT_TEMPLATES
    else:
        raise invalid_param("language", "unsupported language of prompt templates")

    response = []
    for template in templates:
        response.append(
            view_models.PromptTemplate(
                name=template["name"],
                prompt=template["prompt"],
                description=template["description"],
            )
        )
    return view_models.PromptTemplateList(items=response)
