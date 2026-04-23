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

"""Single-source declarative ``Base`` for every SQLAlchemy model.

Phase 3 domain-split requires per-domain DB modules
(``aperag/domains/<domain>/db/models.py``). Each domain's models must
share the same ``Base.metadata`` so Alembic autogeneration sees one
consistent table registry; forking ``Base`` per domain would split
the metadata and break ``alembic check``.

Extracting ``Base`` here keeps ``aperag.db.base`` free of the G1
strict-ban target list (``aperag.db.models`` / ``aperag.schema.view_models``
/ ``aperag.service.*``), so every ``aperag/domains/**/db/models.py`` can
``from aperag.db.base import Base`` without tripping the domain boundary
test. ``aperag/db/models.py`` re-exports ``Base`` so existing call sites
(including the graphindex models and the Alembic ``env.py``) keep working
unchanged until Phase 6 cleanup.
"""

from __future__ import annotations

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

__all__ = ["Base"]
