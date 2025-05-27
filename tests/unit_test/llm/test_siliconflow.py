import os

import pytest

from aperag.llm.completion_service import CompletionService

MODEL = "Qwen/Qwen2.5-7B-Instruct"
SILICONFLOW_API_BASE = os.getenv("SILICONFLOW_API_BASE")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

# Global marker: each test that uses @SKIP will run only if the key exists
SKIP = pytest.mark.skipif(
    not SILICONFLOW_API_BASE or not SILICONFLOW_API_KEY,
    reason="Environment variable SILICONFLOW_API_KEY or SILICONFLOW_API_BASE is not set",
)


def build_service() -> CompletionService:
    kwargs = {
        "model": MODEL,
        "base_url": SILICONFLOW_API_BASE,
        "api_key": SILICONFLOW_API_KEY,
    }
    return CompletionService(**kwargs)


@SKIP
def test_openai_completion():
    llm = build_service()
    resp = llm.generate_stream(history="", prompt="just print 'foobar', nothing else")
    response = ""
    for tokens in resp:
        response += tokens
    assert response == "foobar"
