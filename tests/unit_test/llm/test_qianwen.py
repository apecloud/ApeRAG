import os

import pytest

from aperag.llm.qianwen import QianWenPredictor

QIANWEN_API_KEY = os.getenv("QIANWEN_API_KEY")

# Global marker: each test that uses @SKIP will run only if the key exists
SKIP = pytest.mark.skipif(
    not QIANWEN_API_KEY,
    reason="Environment variable QIANWEN_API_KEY is not set",
)


def build_service() -> QianWenPredictor:
    kargs = {
        "model": "qwen-turbo",
        "api_key": QIANWEN_API_KEY,
    }
    return QianWenPredictor(**kargs)


@SKIP
def test_completion():
    llm = build_service()
    resp = llm.generate_stream(history="", prompt="just print 'foobar', nothing else")
    response = ""
    for tokens in resp:
        response += tokens
    assert response == "foobar"
