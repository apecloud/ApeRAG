import os

import pytest

from aperag.llm.openai import OpenAIPredictor

MODEL = "gpt-4o-mini"
API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Global marker: each test that uses @SKIP will run only if the key exists
SKIP = pytest.mark.skipif(
    not API_BASE or not OPENAI_KEY,
    reason="Environment variable OPENAI_API_BASE or OPENAI_API_KEY is not set",
)


def build_service() -> OpenAIPredictor:
    kwargs = {
        "model": MODEL,
        "base_url": API_BASE,
        "api_key": OPENAI_KEY,
    }
    return OpenAIPredictor(**kwargs)


@SKIP
def test_completion():
    llm = build_service()
    resp = llm.generate_stream(history="", prompt="just print 'foobar', nothing else")
    response = ""
    for tokens in resp:
        response += tokens
    assert response == "foobar"
