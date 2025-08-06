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
MCP Agent OpenTelemetry Integration

Simplified version that provides automatic trace_id injection for mcp-agent events.
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

# Global patch state
_mcp_tracing_enabled = False


def get_current_trace_info() -> Tuple[Optional[str], Optional[str]]:
    """
    Extract trace_id and span_id from current OpenTelemetry context.

    Returns:
        Tuple of (trace_id, span_id) as hex strings, or (None, None) if unavailable
    """
    if not OPENTELEMETRY_AVAILABLE:
        return None, None

    try:
        current_span = trace.get_current_span()
        if not current_span or not current_span.is_recording():
            return None, None

        span_context = current_span.get_span_context()
        if not span_context or not span_context.is_valid:
            return None, None

        trace_id = format(span_context.trace_id, "032x")
        span_id = format(span_context.span_id, "016x")
        return trace_id, span_id

    except Exception:
        # Fail gracefully to avoid breaking normal operation
        return None, None


def _patched_event_method(self, etype, ename, message, context, data):
    """
    Patched Logger.event method that automatically injects trace context.
    """
    try:
        # Handle session_id (preserve original logic)
        if self.session_id:
            if context is None:
                from mcp_agent.logging.events import EventContext

                context = EventContext(session_id=self.session_id)
            elif context.session_id is None:
                context.session_id = self.session_id

        # Get current trace context
        trace_id, span_id = get_current_trace_info()

        # Create Event with trace context
        from mcp_agent.logging.events import Event

        evt = Event(
            type=etype,
            name=ename,
            namespace=self.namespace,
            message=message,
            context=context,
            data=data or {},
            trace_id=trace_id,
            span_id=span_id,
        )

        self._emit_event(evt)
    except Exception as e:
        logger.warning(f"Failed to emit mcp-agent event with trace injection: {e}")

        # Try to call the original method as fallback
        try:
            if hasattr(self, "_original_event"):
                self._original_event(etype, ename, message, context, data or {})
            else:
                logger.info(f"mcp-agent event (fallback): {etype} - {message}")
        except Exception:
            pass  # Silently ignore fallback failures


def init_mcp_tracing() -> bool:
    """
    Initialize MCP agent trace injection.

    Returns:
        True if tracing was enabled, False otherwise
    """
    global _mcp_tracing_enabled

    if _mcp_tracing_enabled:
        logger.debug("MCP agent tracing already enabled")
        return True

    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not available - MCP tracing disabled")
        return False

    try:
        from mcp_agent.logging.logger import Logger

        # Store original method for potential restoration
        if not hasattr(Logger, "_original_event"):
            Logger._original_event = Logger.event

        # Apply the patch
        Logger.event = _patched_event_method

        _mcp_tracing_enabled = True
        logger.info("MCP agent trace injection enabled")
        return True

    except ImportError:
        logger.warning("mcp-agent not available - trace injection disabled")
        return False
    except Exception as e:
        logger.warning(f"Failed to enable MCP trace injection: {e}")
        return False


def is_mcp_tracing_enabled() -> bool:
    """Check if MCP trace injection is enabled."""
    return _mcp_tracing_enabled
