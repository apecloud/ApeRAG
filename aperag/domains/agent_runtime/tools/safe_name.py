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

"""SafeToolName resolver -- D9 §A1 + §A6.

Provider-safe canonical wire/at-rest tool names are required because
many model providers (OpenAI function-calling, Anthropic tool_use,
Vercel AI SDK v5 tool spec) constrain ``tool.name`` to
``[a-zA-Z0-9_-]``. MCP servers, on the other hand, freely use dotted
namespaces (``aperag-knowledge-base.search_collection``,
``user.notes/create_note``). The wire/at-rest part type literal must
hold the safe form; the raw MCP identity is preserved in the part
``metadata``.

Design points pinned here:

* :func:`sanitize_tool_name` runs the spec ``[^a-zA-Z0-9_-] -> _``
  substitution and never raises. Empty input is rejected by the
  caller (``SafeNameRegistry.resolve``) rather than silently
  collapsing to an empty string.
* :class:`SafeNameRegistry` keeps the canonical
  ``(mcp_server, mcp_tool_name) <-> safe_name`` mapping in both
  directions so the runtime can:
    1. Normalise an MCP-side identity into a safe wire name once and
       reuse it across the turn (stable safe names per D9 §A6).
    2. Recover the raw MCP identity from a wire-side ``tool_name``
       (e.g. when the FE consent UI POSTs back a decision keyed on
       the safe name -- the runtime must know which MCP server +
       tool to dispatch to).
* Collisions are resolved by appending ``__<sha256[:6]>`` per the D9
  §A6 hash suffix rule. The hash input is always
  ``f"{mcp_server}|{mcp_tool_name}"`` -- stable across processes and
  consistent with the ``hash`` produced by other ApeRAG layers that
  audit-log raw MCP identities.

The registry is intentionally synchronous and non-IO: callers (the
runtime + the future ``aperag/domains/mcp/`` registry, D9 §1.2)
populate it from their own DB / cache layer at turn start. The
runtime does not need to await on safe-name resolution mid-stream.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from typing import Optional

# D9 §A1 / §A6: provider-safe character class. Re-exported so callers
# (test suites + future MCP registry) share one regex source.
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")

_HASH_SUFFIX_LENGTH = 6


def sanitize_tool_name(raw: str) -> str:
    """Substitute every disallowed character with ``_``.

    Per D9 §A6 the substitution is character-wise -- runs of
    illegal characters are NOT collapsed (``"a..b"`` becomes
    ``"a__b"``, not ``"a_b"``). Preserving length keeps the hash
    suffix deterministic on collision and matches the spec example.
    """

    return SAFE_NAME_RE.sub("_", raw)


def _hash_suffix(mcp_server: str, mcp_tool_name: str) -> str:
    """Stable 6-hex-char sha256 prefix of ``"{server}|{tool}"``.

    Stable across processes / replays / re-registers. The double
    underscore separator (``"__"``) used by callers acts as the
    visual marker of a hash-disambiguated name (per D9 §A6).
    """

    digest = hashlib.sha256(f"{mcp_server}|{mcp_tool_name}".encode("utf-8")).hexdigest()
    return digest[:_HASH_SUFFIX_LENGTH]


@dataclass(frozen=True)
class SafeToolNameResult:
    """Outcome of resolving an ``(mcp_server, mcp_tool_name)`` pair.

    ``safe_name`` is the wire / at-rest provider-safe form;
    ``collided`` is ``True`` iff the naive sanitised form was already
    bound to a different MCP identity and the hash suffix kicked in
    -- callers can audit-log on collision to surface namespace
    pressure.
    """

    safe_name: str
    mcp_server: str
    mcp_tool_name: str
    collided: bool


class SafeNameRegistry:
    """Bidirectional ``(mcp_server, mcp_tool_name) <-> safe_name`` map.

    Thread-safe: a single :class:`threading.Lock` guards mutation;
    reads are lock-free against the latest dict snapshot. Concurrent
    ``resolve`` of the same pair is idempotent -- the second caller
    sees the same ``safe_name``.

    The registry does NOT cross turn boundaries by itself: callers
    that need per-turn isolation construct a fresh instance. The
    runtime / MCP registry typically wants a shared, process-lifetime
    registry so safe names stay stable across turns.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # naive_safe_name -> (mcp_server, mcp_tool_name)
        self._safe_to_mcp: dict[str, tuple[str, str]] = {}
        # (mcp_server, mcp_tool_name) -> safe_name
        self._mcp_to_safe: dict[tuple[str, str], str] = {}

    def resolve(self, mcp_server: str, mcp_tool_name: str) -> SafeToolNameResult:
        """Return the canonical safe name for ``(server, tool)``.

        Raises ``ValueError`` when either input is empty -- empty
        identifiers cannot round-trip through MCP and would produce
        an ambiguous ``"_"``-only safe name.
        """

        if not mcp_server or not mcp_tool_name:
            raise ValueError("mcp_server and mcp_tool_name must be non-empty")

        key = (mcp_server, mcp_tool_name)
        existing = self._mcp_to_safe.get(key)
        if existing is not None:
            return SafeToolNameResult(
                safe_name=existing,
                mcp_server=mcp_server,
                mcp_tool_name=mcp_tool_name,
                # The collision flag tracks whether the original
                # resolution had to disambiguate; a re-lookup of an
                # already-collided pair still reports ``True``.
                collided=existing.endswith("__" + _hash_suffix(mcp_server, mcp_tool_name)),
            )

        naive = sanitize_tool_name(f"{mcp_server}_{mcp_tool_name}")
        with self._lock:
            existing = self._mcp_to_safe.get(key)
            if existing is not None:
                # Lost the race; reuse the winner's safe name.
                return SafeToolNameResult(
                    safe_name=existing,
                    mcp_server=mcp_server,
                    mcp_tool_name=mcp_tool_name,
                    collided=existing.endswith("__" + _hash_suffix(mcp_server, mcp_tool_name)),
                )

            owner = self._safe_to_mcp.get(naive)
            if owner is None or owner == key:
                self._safe_to_mcp[naive] = key
                self._mcp_to_safe[key] = naive
                return SafeToolNameResult(
                    safe_name=naive,
                    mcp_server=mcp_server,
                    mcp_tool_name=mcp_tool_name,
                    collided=False,
                )

            # Collision: another (server, tool) pair already claimed
            # the naive form. Append the stable hash suffix.
            suffix = _hash_suffix(mcp_server, mcp_tool_name)
            qualified = f"{naive}__{suffix}"
            self._safe_to_mcp[qualified] = key
            self._mcp_to_safe[key] = qualified
            return SafeToolNameResult(
                safe_name=qualified,
                mcp_server=mcp_server,
                mcp_tool_name=mcp_tool_name,
                collided=True,
            )

    def reverse(self, safe_name: str) -> Optional[tuple[str, str]]:
        """Look up ``(mcp_server, mcp_tool_name)`` for a wire safe name.

        Returns ``None`` when the safe name is unknown. Callers must
        treat ``None`` as "tool not registered" rather than retrying.
        """

        return self._safe_to_mcp.get(safe_name)

    def metadata(self, safe_name: str) -> dict[str, str]:
        """Metadata dict suitable for the ``tool-<name>`` part.

        Empty dict for unknown safe names so callers can blindly
        spread it onto the wire frame without a guard. Populated
        keys are ``mcpServer`` / ``mcpToolName`` (per D9 §A1 wire
        shape).
        """

        match = self.reverse(safe_name)
        if match is None:
            return {}
        server, tool = match
        return {"mcpServer": server, "mcpToolName": tool}

    def known_safe_names(self) -> list[str]:
        """Snapshot of registered safe names (for tests / audit)."""

        return list(self._safe_to_mcp.keys())


__all__ = [
    "SAFE_NAME_RE",
    "SafeNameRegistry",
    "SafeToolNameResult",
    "sanitize_tool_name",
]
