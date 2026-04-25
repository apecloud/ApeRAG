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

"""MCP server registry -- D9 §1.1 + §A5.

Three-tier registry (system / bot / user) with the D9 hard
invariants:

* **System namespace is reserved.** A user / bot ``register`` whose
  ``name`` collides with a system entry is rejected with
  :class:`RegistryConflictError`. There is no silent shadow.
* **No silent override.** Even when admin policy intentionally
  shadows a system tool with a user / bot alias, the override goes
  through :meth:`MCPServerRegistry.register_admin_alias` which
  audit-logs every change. There is no public ``register`` flag that
  silently bypasses the conflict check.
* **Effective resolution preserves system precedence.** When the
  runtime asks :meth:`MCPServerRegistry.effective_servers` for the
  list of servers visible to a particular ``(user, bot)`` invocation,
  iteration order is ``system -> bot -> user``; same-name entries in
  later tiers are skipped (per D9 §1.1 step 3, "no override").

This module is the in-process registry interface that the future
``aperag/domains/mcp/`` DB-backed registry (D9 §1.2) will satisfy. We
intentionally do not own DB or alembic surface here -- D8.3 / #75 is
backend lifecycle behaviour, not a new domain. Callers wire DB-loaded
entries into the registry at startup.

Audit log entries are emitted via a callable injected at
construction; tests use an in-memory list, production wires the
existing ``audit_log_service``. Keeping the audit sink behind an
interface keeps this module DB-free and trivially testable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

# -- audit sink protocol ------------------------------------------------


AuditLogger = Callable[[str, dict[str, Any]], None]
"""``(event_name, payload) -> None`` audit-sink callable.

Production wires ``aperag.audit.audit_log_service.record_audit``;
tests pass an in-memory ``list.append`` lambda.
"""


# -- enums + dataclasses ------------------------------------------------


class RegistryScope(str, Enum):
    """D9 §1.1 three-tier scope."""

    SYSTEM = "system"
    BOT = "bot"
    USER = "user"


@dataclass(frozen=True)
class MCPServerEntry:
    """Single MCP server registration.

    ``scope_ref`` carries the ``bot_id`` for ``bot`` scope and the
    ``user_id`` for ``user`` scope; for ``system`` scope it is
    ``None`` (system entries apply to all callers). The dataclass is
    frozen so callers can hash entries into sets without surprise
    aliasing.
    """

    scope: RegistryScope
    name: str
    url: str
    scope_ref: Optional[str] = None
    enabled: bool = True
    auth_config: tuple[tuple[str, Any], ...] = ()
    """``auth_config`` is stored as a tuple of pairs (frozen-friendly
    form of an immutable mapping). Use :meth:`auth_config_dict` for a
    convenience copy."""

    def auth_config_dict(self) -> dict[str, Any]:
        return dict(self.auth_config)


# -- exceptions ---------------------------------------------------------


class RegistryConflictError(RuntimeError):
    """Raised on a non-admin attempt to shadow ``system`` namespace.

    D9 §1.1 / §A5: user / bot may NOT silently register a tool with a
    name that collides with a system entry. The runtime surfaces this
    error to the registration call site (typically a per-bot config
    validator) and the caller decides whether to reject the bot
    config or surface a clear UX message. Admin-mediated alias goes
    through :meth:`MCPServerRegistry.register_admin_alias`.
    """


# -- registry -----------------------------------------------------------


@dataclass
class _ScopeIndex:
    """Internal: name-keyed view of one tier."""

    entries: dict[str, MCPServerEntry] = field(default_factory=dict)


class MCPServerRegistry:
    """In-process MCP server registry with D9 §1.1+§A5 invariants.

    Construct with an :class:`AuditLogger` and feed entries via
    :meth:`register`; admin shadow alias goes through
    :meth:`register_admin_alias`. ``effective_servers`` returns the
    resolved list for a ``(user, bot)`` invocation and is the primary
    consumer surface for the agent runtime.
    """

    def __init__(self, *, audit_logger: Optional[AuditLogger] = None):
        self._lock = threading.Lock()
        self._audit_logger = audit_logger or _noop_audit
        self._tiers: dict[RegistryScope, _ScopeIndex] = {
            RegistryScope.SYSTEM: _ScopeIndex(),
            RegistryScope.BOT: _ScopeIndex(),
            RegistryScope.USER: _ScopeIndex(),
        }
        # admin aliases keyed on ``(scope, scope_ref, name)`` so a bot
        # can intentionally shadow a system tool only after an admin
        # action -- the alias survives independently of the regular
        # registration so audit trail is preserved even on re-register.
        self._admin_aliases: dict[tuple[RegistryScope, Optional[str], str], MCPServerEntry] = {}

    # -- public registration ---------------------------------------

    def register(self, entry: MCPServerEntry) -> None:
        """Register ``entry`` in its scope tier.

        Raises :class:`RegistryConflictError` when a non-system
        ``entry`` collides with an existing system entry of the same
        name -- per D9 §A5 there is no silent override path here.
        Admin override goes through :meth:`register_admin_alias`.

        Re-registration of an existing key is permitted (idempotent
        update of url/auth/enabled); it is audit-logged so stale-cache
        replay is debuggable.
        """

        with self._lock:
            if entry.scope is not RegistryScope.SYSTEM:
                system_clash = self._tiers[RegistryScope.SYSTEM].entries.get(entry.name)
                if system_clash is not None:
                    self._audit("registry.system_namespace_rejected", _entry_audit_payload(entry))
                    raise RegistryConflictError(
                        f"Cannot register {entry.scope.value} server '{entry.name}': "
                        "name is reserved by the system namespace."
                    )
            self._tiers[entry.scope].entries[entry.name] = entry
            self._audit("registry.registered", _entry_audit_payload(entry))

    def unregister(self, scope: RegistryScope, name: str) -> bool:
        """Remove ``name`` from ``scope``. Returns ``True`` if removed.

        Idempotent on missing entries (returns ``False``). Audit-logged
        on success so the trail captures intentional removals.
        """

        with self._lock:
            removed = self._tiers[scope].entries.pop(name, None) is not None
            if removed:
                self._audit("registry.unregistered", {"scope": scope.value, "name": name})
            return removed

    def register_admin_alias(
        self,
        *,
        target_scope: RegistryScope,
        target_scope_ref: Optional[str],
        alias: MCPServerEntry,
        admin_user_id: str,
        reason: str,
    ) -> None:
        """Admin-mediated explicit override of a name in ``target_scope``.

        D9 §A5 / §1.1 step 4: admin override is the ONLY path to
        shadow a system namespace name. Every call is audit-logged
        with the admin identity + reason so the trail makes the
        intent reviewable.

        ``alias.scope`` should match ``target_scope`` -- the call
        rejects mismatches because an alias outside the override
        scope is a configuration bug.
        """

        if alias.scope is not target_scope:
            raise ValueError(f"alias.scope ({alias.scope.value}) must match target_scope ({target_scope.value})")
        with self._lock:
            key = (target_scope, target_scope_ref, alias.name)
            self._admin_aliases[key] = alias
            self._tiers[target_scope].entries[alias.name] = alias
            self._audit(
                "registry.admin_alias_registered",
                {
                    **_entry_audit_payload(alias),
                    "target_scope_ref": target_scope_ref,
                    "admin_user_id": admin_user_id,
                    "reason": reason,
                },
            )

    # -- resolution --------------------------------------------------

    def effective_servers(self, *, user_id: str, bot_id: Optional[str]) -> list[MCPServerEntry]:
        """Resolve the server list visible to a ``(user, bot)`` turn.

        Resolution order is ``system -> bot -> user``; later tiers
        skip entries whose name is already bound by an earlier tier
        (per D9 §1.1 step 3, "no override"). ``enabled=False``
        entries are filtered out at the resolution boundary so the
        runtime never sees a disabled server.
        """

        seen_names: set[str] = set()
        out: list[MCPServerEntry] = []

        for entry in self._tiers[RegistryScope.SYSTEM].entries.values():
            if entry.enabled:
                seen_names.add(entry.name)
                out.append(entry)

        if bot_id is not None:
            for entry in self._tiers[RegistryScope.BOT].entries.values():
                if not entry.enabled:
                    continue
                if entry.scope_ref is not None and entry.scope_ref != bot_id:
                    continue
                if entry.name in seen_names:
                    continue
                seen_names.add(entry.name)
                out.append(entry)

        for entry in self._tiers[RegistryScope.USER].entries.values():
            if not entry.enabled:
                continue
            if entry.scope_ref is not None and entry.scope_ref != user_id:
                continue
            if entry.name in seen_names:
                continue
            seen_names.add(entry.name)
            out.append(entry)

        return out

    # -- introspection (for tests + admin views) --------------------

    def list_in_scope(self, scope: RegistryScope) -> list[MCPServerEntry]:
        return list(self._tiers[scope].entries.values())

    def list_admin_aliases(self) -> list[tuple[RegistryScope, Optional[str], str, MCPServerEntry]]:
        return [(scope, scope_ref, name, entry) for (scope, scope_ref, name), entry in self._admin_aliases.items()]

    # -- audit helper ------------------------------------------------

    def _audit(self, event: str, payload: dict[str, Any]) -> None:
        try:
            self._audit_logger(event, payload)
        except Exception:  # pragma: no cover - audit must never break registration
            # Audit is best-effort; registry mutation already
            # happened. Production audit sinks must absorb their own
            # errors, this guard exists so test fakes don't crash
            # registry call sites if the lambda raises.
            pass


# -- internals ----------------------------------------------------------


def _noop_audit(event: str, payload: dict[str, Any]) -> None:  # pragma: no cover - trivial default
    return None


def _entry_audit_payload(entry: MCPServerEntry) -> dict[str, Any]:
    return {
        "scope": entry.scope.value,
        "scope_ref": entry.scope_ref,
        "name": entry.name,
        "url": entry.url,
        "enabled": entry.enabled,
    }


__all__ = [
    "AuditLogger",
    "MCPServerEntry",
    "MCPServerRegistry",
    "RegistryConflictError",
    "RegistryScope",
]
