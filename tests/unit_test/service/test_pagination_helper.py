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

"""D10.e (#97) follow-up — :mod:`aperag.service.pagination` helper tests.

Pins the canonical-error contract that flows out of every D10 list
primitive when its caller ships a malformed / scope-mismatched cursor.
Each list primitive (``list_collections`` / ``list_documents``) calls
:func:`decode_offset_cursor` exactly once per request, so this helper
is the §C.3 chokepoint that determines what error code reaches the
client.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import pytest

from aperag.mcp.cursor import CursorPayload, encode_cursor
from aperag.mcp.cursor.codec import CURSOR_SCHEMA_VERSION
from aperag.mcp.cursor.errors import CursorError
from aperag.service.pagination import (
    decode_offset_cursor,
    encode_offset_cursor,
)


def _scope(**overrides: Any) -> dict[str, Any]:
    base = dict(
        sort_key="created_at",
        filters={"title_filter": None, "sort_order": "desc"},
        collection_id=None,
        tenant_id="tnt-1",
    )
    base.update(overrides)
    return base


class TestNoneAndEmpty:
    def test_none_cursor_returns_zero(self):
        assert decode_offset_cursor(None, **_scope()) == 0

    def test_empty_cursor_returns_zero(self):
        assert decode_offset_cursor("", **_scope()) == 0


class TestRoundTrip:
    def test_round_trip_offset(self):
        token = encode_offset_cursor(offset=50, **_scope())
        assert decode_offset_cursor(token, **_scope()) == 50

    def test_round_trip_zero_offset(self):
        token = encode_offset_cursor(offset=0, **_scope())
        assert decode_offset_cursor(token, **_scope()) == 0


class TestMalformed:
    def test_garbage_string_raises_cursor_invalid(self):
        with pytest.raises(CursorError) as excinfo:
            decode_offset_cursor("not~~base64??", **_scope())
        assert excinfo.value.code == "cursor_invalid"

    def test_valid_base64_but_not_json_raises_cursor_invalid(self):
        token = base64.urlsafe_b64encode(b"not json at all").rstrip(b"=").decode("ascii")
        with pytest.raises(CursorError) as excinfo:
            decode_offset_cursor(token, **_scope())
        assert excinfo.value.code == "cursor_invalid"


class TestScopeMismatch:
    def _payload_token(self, **overrides: Any) -> str:
        from aperag.mcp.cursor.invariants import compute_invariant_hash

        scope = _scope(**overrides)
        invariant = compute_invariant_hash(
            sort_key=scope["sort_key"],
            filters=scope["filters"],
            collection_id=scope["collection_id"],
            tenant_id=scope["tenant_id"],
        )
        payload = CursorPayload(
            sort_key=scope["sort_key"],
            last_position={"offset": 30},
            invariant_hash=invariant,
            issued_at=int(time.time()),
            server_id="srv-test",
        )
        return encode_cursor(payload)

    def test_sort_key_changed_raises_cursor_filter_mismatch(self):
        token = self._payload_token(sort_key="created_at")
        with pytest.raises(CursorError) as excinfo:
            decode_offset_cursor(token, **_scope(sort_key="title"))
        assert excinfo.value.code == "cursor_filter_mismatch"

    def test_filters_changed_raises_cursor_filter_mismatch(self):
        token = self._payload_token(
            filters={"title_filter": None, "sort_order": "desc"},
        )
        with pytest.raises(CursorError) as excinfo:
            decode_offset_cursor(
                token,
                **_scope(filters={"title_filter": "needle", "sort_order": "desc"}),
            )
        assert excinfo.value.code == "cursor_filter_mismatch"

    def test_collection_id_changed_raises_cursor_filter_mismatch(self):
        token = self._payload_token(collection_id="col-A")
        with pytest.raises(CursorError) as excinfo:
            decode_offset_cursor(token, **_scope(collection_id="col-B"))
        assert excinfo.value.code == "cursor_filter_mismatch"


class TestExpired:
    def test_expired_cursor_raises_cursor_expired(self):
        from aperag.mcp.cursor.invariants import compute_invariant_hash

        scope = _scope()
        invariant = compute_invariant_hash(
            sort_key=scope["sort_key"],
            filters=scope["filters"],
            collection_id=scope["collection_id"],
            tenant_id=scope["tenant_id"],
        )
        # Issue with an issued_at far in the past so the codec's
        # expiry check fires before the helper's own scope check.
        payload = CursorPayload(
            sort_key=scope["sort_key"],
            last_position={"offset": 10},
            invariant_hash=invariant,
            issued_at=1000,
            ttl_seconds=60,
            server_id="srv-test",
        )
        token = encode_cursor(payload)
        with pytest.raises(CursorError) as excinfo:
            decode_offset_cursor(token, **scope)
        assert excinfo.value.code == "cursor_expired"


class TestSchemaVersion:
    def test_unknown_schema_version_raises_cursor_schema_unsupported(self):
        from aperag.mcp.cursor.invariants import compute_invariant_hash

        scope = _scope()
        invariant = compute_invariant_hash(
            sort_key=scope["sort_key"],
            filters=scope["filters"],
            collection_id=scope["collection_id"],
            tenant_id=scope["tenant_id"],
        )
        payload = CursorPayload(
            schema_version=CURSOR_SCHEMA_VERSION + 99,
            sort_key=scope["sort_key"],
            last_position={"offset": 10},
            invariant_hash=invariant,
            issued_at=int(time.time()),
            server_id="srv-test",
        )
        token = encode_cursor(payload)
        with pytest.raises(CursorError) as excinfo:
            decode_offset_cursor(token, **scope)
        assert excinfo.value.code == "cursor_schema_unsupported"


class TestLastPositionInvariants:
    """``last_position.offset`` itself must be a non-negative int."""

    def _token_with_offset(self, raw_offset: Any) -> str:
        from aperag.mcp.cursor.invariants import compute_invariant_hash

        scope = _scope()
        invariant = compute_invariant_hash(
            sort_key=scope["sort_key"],
            filters=scope["filters"],
            collection_id=scope["collection_id"],
            tenant_id=scope["tenant_id"],
        )
        # We bypass the strongly-typed CursorPayload to craft a
        # malformed last_position payload — the codec is structural
        # and the rejection happens inside the helper.
        raw = json.dumps(
            {
                "schema_version": CURSOR_SCHEMA_VERSION,
                "sort_key": scope["sort_key"],
                "last_position": {"offset": raw_offset},
                "invariant_hash": invariant,
                "issued_at": int(time.time()),
                "ttl_seconds": 3600,
                "server_id": "srv-test",
                "extra": {},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    def test_string_offset_raises_cursor_invalid(self):
        with pytest.raises(CursorError) as excinfo:
            decode_offset_cursor(self._token_with_offset("ten"), **_scope())
        assert excinfo.value.code == "cursor_invalid"

    def test_negative_offset_raises_cursor_invalid(self):
        with pytest.raises(CursorError) as excinfo:
            decode_offset_cursor(self._token_with_offset(-1), **_scope())
        assert excinfo.value.code == "cursor_invalid"

    def test_bool_offset_raises_cursor_invalid(self):
        # bool is a subclass of int in Python; the helper must reject
        # it explicitly so ``True`` doesn't slip through as offset 1.
        with pytest.raises(CursorError) as excinfo:
            decode_offset_cursor(self._token_with_offset(True), **_scope())
        assert excinfo.value.code == "cursor_invalid"
