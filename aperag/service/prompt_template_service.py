from http import HTTPStatus

from aperag.llm.prompts import MULTI_ROLE_EN_PROMPT_TEMPLATES, MULTI_ROLE_ZH_PROMPT_TEMPLATES
from aperag.schema import view_models
from aperag.views.utils import fail, success


def list_prompt_templates(language: str) -> view_models.PromptTemplateList:
    if language == "zh-CN":
        templates = MULTI_ROLE_ZH_PROMPT_TEMPLATES
    elif language == "en-US":
        templates = MULTI_ROLE_EN_PROMPT_TEMPLATES
    else:
        return fail(HTTPStatus.BAD_REQUEST, "unsupported language of prompt templates")
    response = []
    for template in templates:
        response.append(
            view_models.PromptTemplate(
                name=template["name"],
                prompt=template["prompt"],
                description=template["description"],
            )
        )
    return success(view_models.PromptTemplateList(items=response))
