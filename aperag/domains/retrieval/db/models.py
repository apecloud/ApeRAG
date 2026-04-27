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

"""Retrieval-domain SQLAlchemy models.

Owns ``SearchHistory``, the per-user audit row for ``/search``
invocations (text + typed recall payloads + surfaced items). The
table body was moved here from ``aperag.db.models`` in Phase 3 step 3
as part of the retrieval domain split; the legacy aggregate module
keeps a re-export shim until Phase 6 cleanup.
"""

from __future__ import annotations

import random
import uuid

from sqlalchemy import JSON, Column, DateTime, String, Text

from aperag.db.base import Base
from aperag.utils.utils import utc_now


def _random_id() -> str:
    """Local copy of ``aperag.db.models.random_id`` so the retrieval db
    module does not have to import from the G1 strict-ban target
    aggregate. Phase 6 cleanup collapses these helper twins.
    """

    return "".join(random.sample(uuid.uuid4().hex, 16))


class SearchHistory(Base):
    __tablename__ = "searchhistory"

    id = Column(String(24), primary_key=True, default=lambda: "sh" + _random_id())
    user = Column(String(256), nullable=False, index=True)  # Add index for user queries
    collection_id = Column(String(24), nullable=True, index=True)  # Add index for collection queries
    query = Column(Text, nullable=False)
    vector_search = Column(JSON, default=lambda: {}, nullable=True)
    fulltext_search = Column(JSON, default=lambda: {}, nullable=True)
    graph_search = Column(JSON, default=lambda: {}, nullable=True)
    summary_search = Column(JSON, default=lambda: {}, nullable=True)
    vision_search = Column(JSON, default=lambda: {}, nullable=True)
    items = Column(JSON, default=lambda: [], nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)  # Add index for soft delete queries


__all__ = ["SearchHistory"]
