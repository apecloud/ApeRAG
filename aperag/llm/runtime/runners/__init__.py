from aperag.llm.runtime.model_runner import runner_registry
from aperag.llm.runtime.runners.dashscope_rerank import DashScopeRerankRunner
from aperag.llm.runtime.runners.jina_rerank import JinaRerankRunner
from aperag.llm.runtime.runners.litellm import LiteLLMRunner
from aperag.llm.runtime.runners.openai_compatible import OpenAICompatibleRunner

runner_registry.register("openai_compatible", OpenAICompatibleRunner())
runner_registry.register("dashscope_rerank", DashScopeRerankRunner())
runner_registry.register("jina_rerank", JinaRerankRunner())
runner_registry.register("litellm", LiteLLMRunner())

__all__ = ["runner_registry"]
