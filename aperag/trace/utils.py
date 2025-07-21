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

"""
Tracing utility functions.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# OpenTelemetry imports with graceful fallback
try:
    from opentelemetry import trace

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False


def get_tracer(name: str):
    """
    Get a tracer instance.

    Args:
        name: Tracer name (usually module name)

    Returns:
        Tracer instance or None if OpenTelemetry is not available
    """
    if OPENTELEMETRY_AVAILABLE:
        return trace.get_tracer(name)
    return None


def get_current_trace_info() -> Tuple[Optional[str], Optional[str]]:
    """
    Get current trace and span IDs.

    Returns:
        Tuple of (trace_id, span_id) as hex strings, or (None, None) if no active span
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None, None

    try:
        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().is_valid:
            span_context = current_span.get_span_context()
            trace_id = format(span_context.trace_id, "032x")
            span_id = format(span_context.span_id, "016x")
            return trace_id, span_id
        return None, None
    except Exception:
        return None, None


def create_span(tracer, name: str, **attributes):
    """
    Create a new span with the given attributes.

    Args:
        tracer: Tracer instance (can be None)
        name: Span name
        **attributes: Span attributes

    Returns:
        Span context manager or a no-op context manager
    """
    if tracer is None or not OPENTELEMETRY_AVAILABLE:
        # Return a no-op context manager
        from contextlib import nullcontext

        return nullcontext()

    try:
        # Create a wrapper that handles attribute setting properly
        class SpanWrapper:
            def __init__(self, span):
                self.span = span

            def __enter__(self):
                span = self.span.__enter__()
                # Set attributes after entering the span context
                for key, value in attributes.items():
                    try:
                        span.set_attribute(key, value)
                    except Exception:
                        pass  # Ignore attribute setting failures
                return span

            def __exit__(self, exc_type, exc_val, exc_tb):
                return self.span.__exit__(exc_type, exc_val, exc_tb)

        span = tracer.start_as_current_span(name)
        return SpanWrapper(span)
    except Exception as e:
        logger.warning(f"Failed to create span '{name}': {e}")
        from contextlib import nullcontext

        return nullcontext()


def trace_function(name: Optional[str] = None):
    """
    Decorator to automatically trace a function.

    Args:
        name: Custom span name (defaults to function name)

    Usage:
        @trace_function()
        def my_function():
            pass

        @trace_function("custom_name")
        def my_function():
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            span_name = name or f"{func.__name__}"

            with create_span(tracer, span_name, function=func.__name__, module=func.__module__):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_async_function(name: Optional[str] = None, new_trace: bool = False):
    """
    Decorator to automatically trace an async function.

    Args:
        name: Custom span name (defaults to function name)
        new_trace: If True, creates a new trace (new trace_id) instead of a child span

    Usage:
        @trace_async_function()
        async def my_async_function():
            pass

        @trace_async_function("custom_name", new_trace=True)
        async def my_async_function():
            pass
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            span_name = name or f"{func.__qualname__}"

            if new_trace:
                # Create a new trace (root span) instead of child span
                with create_new_trace_span(tracer, span_name, function=func.__name__, module=func.__module__):
                    return await func(*args, **kwargs)
            else:
                # Create child span (default behavior)
                with create_span(tracer, span_name, function=func.__name__, module=func.__module__):
                    return await func(*args, **kwargs)

        return wrapper

    return decorator


def create_new_trace_span(tracer, name: str, **attributes):
    """
    Create a new root span that starts a new trace (new trace_id).

    Args:
        tracer: Tracer instance (can be None)
        name: Span name
        **attributes: Span attributes

    Returns:
        Span context manager or a no-op context manager
    """
    if tracer is None or not OPENTELEMETRY_AVAILABLE:
        # Return a no-op context manager
        from contextlib import nullcontext

        return nullcontext()

    try:
        from opentelemetry import context
        from opentelemetry.trace import set_span_in_context

        # Create a wrapper that handles attribute setting and context management properly
        class NewTraceSpanWrapper:
            def __init__(self, tracer, span_name, attributes):
                self.tracer = tracer
                self.span_name = span_name
                self.attributes = attributes
                self.span = None
                self.token = None

            def __enter__(self):
                # Start a new span with an empty context (no parent) to create a new trace
                # This ensures we get a new trace_id instead of inheriting from any parent
                empty_context = context.Context()

                # Start span without any parent context
                self.span = self.tracer.start_span(name=self.span_name, context=empty_context)

                # Set attributes after creating the span
                for key, value in self.attributes.items():
                    try:
                        self.span.set_attribute(key, value)
                    except Exception:
                        pass  # Ignore attribute setting failures

                # Make this span the current active span
                span_context = set_span_in_context(self.span, empty_context)
                self.token = context.attach(span_context)

                return self.span

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Restore previous context
                if self.token is not None:
                    context.detach(self.token)

                # End the span
                if self.span:
                    self.span.end()

                return False

        return NewTraceSpanWrapper(tracer, name, attributes)

    except Exception as e:
        logger.warning(f"Failed to create new trace span '{name}': {e}")
        from contextlib import nullcontext

        return nullcontext()


def add_trace_attributes(**attributes):
    """
    Add attributes to the current span.

    Args:
        **attributes: Attributes to add
    """
    if not OPENTELEMETRY_AVAILABLE:
        return

    try:
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
    except Exception:
        pass  # Silently ignore failures
