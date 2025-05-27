import os

import pytest

from aperag.llm.completion_service import CompletionService

MODEL = "gemini/gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Global marker: each test that uses @SKIP will run only if the key exists
SKIP = pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason="Environment variable GEMINI_API_KEY is not set",
)


def build_service() -> CompletionService:
    kwargs = {
        "model": MODEL,
        "base_url": "",
        "api_key": GEMINI_API_KEY,
    }
    return CompletionService(**kwargs)


@SKIP
def test_openai_completion():
    llm = build_service()
    resp = llm.generate_stream(history="", prompt="""just print 'foobar', nothing else""")
    response = ""
    for tokens in resp:
        response += tokens
    assert response.__contains__("foobar")
