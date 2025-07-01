"""
Shared fixtures and utilities for graph storage testing.
Provides common test data and helper functions for all graph storage test suites.
"""

# Note: For pytest-asyncio class-scoped fixtures to work properly,
# tests should use @pytest.mark.asyncio(loop_scope="class") decorator
# instead of custom event_loop fixtures.

import asyncio

import pytest


@pytest.fixture(scope="class")
def event_loop():
    """Create an event loop for class-scoped async fixtures.

    This allows class-scoped fixtures like neo4j_oracle_storage to work with async operations.
    The default event_loop fixture is function-scoped, which causes ScopeMismatch errors
    when accessed by class-scoped fixtures.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()
