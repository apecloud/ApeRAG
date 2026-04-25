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

"""D10.e (#97) cursor contract — codec / invariants / errors / schemas.

Test surface kept narrow so each cursor invariant from design pack
§C.2 lands as a discrete failing test before any read primitive
(D10.c) wires the encoder in. Once D10.c integration arrives, the
e2e hurl suite covers cross-tool flow; these tests stay as the
unit-level pin.
"""

from __future__ import annotations

import pytest

from aperag.mcp.cursor import (
    CursorPayload,
    PaginationParams,
    PaginationResult,
    compute_invariant_hash,
    decode_cursor,
    encode_cursor,
)
from aperag.mcp.cursor.codec import (
    CURSOR_SCHEMA_VERSION,
    DEFAULT_TTL_SECONDS,
    _decode_cursor_payload,
)
from aperag.mcp.cursor.errors import (
    SILENT_RESET_FORBIDDEN,
    CursorError,
    CursorErrorEnvelope,
)


def _payload(**overrides) -> CursorPayload:
    # ``issued_at`` defaults to "now" so round-trip tests don't trip
    # the §C.4 TTL window when the test is run far from the fixture's
    # original drafting date. Tests that need a frozen / expired
    # payload override ``issued_at`` explicitly.
    import time as _time

    base = dict(
        sort_key="created_at",
        last_position={"created_at": "2026-04-26T03:00:00Z", "id": "doc-42"},
        invariant_hash="a" * 64,
        issued_at=int(_time.time()),
        server_id="srv-test",
    )
    base.update(overrides)
    return CursorPayload(**base)


class TestCodec:
    def test_round_trip_preserves_all_fields(self):
        original = _payload(extra={"score_boundary": 0.42})
        token = encode_cursor(original)
        restored = decode_cursor(token)
        assert restored == original

    def test_wire_format_is_url_safe_unpadded(self):
        token = encode_cursor(_payload())
        assert "=" not in token
        assert "+" not in token
        assert "/" not in token

    def test_default_schema_version_and_ttl_match_spec(self):
        payload = _payload()
        assert payload.schema_version == CURSOR_SCHEMA_VERSION == 1
        assert payload.ttl_seconds == DEFAULT_TTL_SECONDS == 3600

    def test_decode_garbage_raises_canonical_cursor_invalid(self):
        # The decoder is the single chokepoint — every caller
        # downstream sees `cursor_invalid` for a malformed wire
        # without each tool reinventing the mapping.
        with pytest.raises(CursorError) as excinfo:
            decode_cursor("not~~base64??")
        assert excinfo.value.code == "cursor_invalid"

    def test_decode_unknown_schema_version_raises_cursor_schema_unsupported(self):
        future_token = encode_cursor(_payload(schema_version=CURSOR_SCHEMA_VERSION + 1))
        with pytest.raises(CursorError) as excinfo:
            decode_cursor(future_token)
        assert excinfo.value.code == "cursor_schema_unsupported"
        assert excinfo.value.details["received_schema_version"] == CURSOR_SCHEMA_VERSION + 1

    def test_decode_past_ttl_raises_cursor_expired(self):
        token = encode_cursor(_payload(issued_at=1000, ttl_seconds=60))
        with pytest.raises(CursorError) as excinfo:
            decode_cursor(token, now=2000)
        assert excinfo.value.code == "cursor_expired"

    def test_is_expired_at_exact_ttl_boundary(self):
        payload = _payload(issued_at=1000, ttl_seconds=60)
        assert payload.is_expired(now=1059) is False
        assert payload.is_expired(now=1060) is True

    def test_internal_decode_skips_schema_and_expiry_checks(self):
        # _decode_cursor_payload is the test/debug escape hatch — it
        # MUST NOT be called from production code, only from tests
        # that need to craft wrong-schema or expired payloads.
        future_token = encode_cursor(_payload(schema_version=999))
        payload = _decode_cursor_payload(future_token)
        assert payload.schema_version == 999


class TestInvariantHash:
    def test_deterministic_across_dict_ordering(self):
        h1 = compute_invariant_hash(
            sort_key="created_at",
            filters={"a": 1, "b": 2},
            collection_id="col-1",
            tenant_id="tnt-1",
        )
        h2 = compute_invariant_hash(
            sort_key="created_at",
            filters={"b": 2, "a": 1},
            collection_id="col-1",
            tenant_id="tnt-1",
        )
        assert h1 == h2

    def test_changing_any_binding_changes_hash(self):
        base = dict(
            sort_key="created_at",
            filters={"a": 1},
            collection_id="col-1",
            tenant_id="tnt-1",
            index_id="idx-1",
        )
        baseline = compute_invariant_hash(**base)

        for field, replacement in (
            ("sort_key", "title"),
            ("filters", {"a": 2}),
            ("collection_id", "col-2"),
            ("tenant_id", "tnt-2"),
            ("index_id", "idx-2"),
        ):
            mutated = {**base, field: replacement}
            assert compute_invariant_hash(**mutated) != baseline, field


class TestErrors:
    @pytest.mark.parametrize(
        "code",
        [
            "cursor_invalid",
            "cursor_expired",
            "cursor_filter_mismatch",
            "cursor_tenant_mismatch",
            "cursor_index_changed",
            "cursor_schema_unsupported",
        ],
    )
    def test_each_canonical_code_round_trips_through_envelope(self, code):
        err = CursorError(code, "boom", details={"hint": "abc"})
        envelope = err.to_envelope()
        assert isinstance(envelope, CursorErrorEnvelope)
        assert envelope.code == code
        assert envelope.message == "boom"
        assert envelope.details == {"hint": "abc"}

    def test_silent_reset_is_forbidden(self):
        # §C.3 anti-pattern guard: servers MUST NOT silently restart
        # pagination on cursor failure. The constant is the visible
        # contract pin — flipping it to False is the loud signal that
        # someone is about to break the explicit-not-silent invariant.
        assert SILENT_RESET_FORBIDDEN is True


class TestPaginationShape:
    def test_default_request_starts_a_fresh_pagination(self):
        params = PaginationParams()
        assert params.cursor is None
        assert params.limit == 50

    def test_limit_ceiling_is_enforced(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PaginationParams(limit=0)
        with pytest.raises(ValidationError):
            PaginationParams(limit=10_000)

    def test_result_generic_carries_typed_items(self):
        from pydantic import BaseModel

        class Item(BaseModel):
            id: str

        result = PaginationResult[Item](items=[Item(id="x")], next_cursor=None)
        assert [it.id for it in result.items] == ["x"]
        assert result.next_cursor is None
        assert result.total_count is None

    def test_round_trip_with_real_cursor(self):
        # End-to-end: encode → wire string → PaginationResult →
        # decode confirms the cursor returned to the client matches
        # what was issued. Simulates the server side of one page
        # boundary without involving any read primitive.
        next_payload = _payload()
        result = PaginationResult[dict](
            items=[{"id": "a"}, {"id": "b"}],
            next_cursor=encode_cursor(next_payload),
            total_count=42,
        )
        assert result.next_cursor is not None
        decoded = decode_cursor(result.next_cursor)
        assert decoded == next_payload
