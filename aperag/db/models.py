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

"""Legacy aggregate ORM module — Phase 8 Task #39 carve aftermath.

After Phase 8 Task #39 every ORM class previously declared here has
either been carved out to a per-domain ``db/models.py`` (Phase 4
identity / governance / model_platform / marketplace + Phase 8 Task #39
``Invitation`` / ``ConfigModel`` / ``UserQuota`` / ``Setting`` /
``ModelServiceProvider`` / ``PromptTemplate`` / ``ExportTask``) or
hard-deleted (the legacy evaluation stack:
``Evaluation`` / ``EvaluationItem`` / ``Question`` / ``QuestionSet``
plus their ``EvaluationStatus`` / ``EvaluationItemStatus`` /
``QuestionType`` enums — superseded by ``aperag.domains.evaluation``).

What's left here is the cross-domain shared utility surface:

* ``Base`` re-export from ``aperag.db.base`` so existing call sites
  (``from aperag.db.models import Base`` — notably
  ``aperag/migration/env.py``, the graphindex test fixtures) keep
  resolving the same declarative base. Phase 9 directory move
  (``aperag/db/`` → ``aperag/platform/db/``) will retire this re-export.
* ``random_id`` / ``EnumColumn`` shared helpers used by per-domain
  ``db/models.py`` modules. Phase 6 cleanup folds the helper twins
  back into ``aperag.db.base``; until then they live here as the
  single canonical implementation.

Note: the Layer C cross-domain ``from aperag.domains.identity.db.models
import Role`` import that existed at module-top before Task #39
naturally dissolved once ``Invitation`` (the only class whose
class-body needed ``Role`` at import time) moved into the identity
domain.
"""

import random
import uuid

from sqlalchemy import String

from aperag.db.base import Base  # noqa: F401  Phase 4 re-export — see module docstring.

__all__ = [
    "Base",
    "EnumColumn",
    "random_id",
]


# Helper function for random id generation
def random_id():
    """Generate a random ID string"""
    return "".join(random.sample(uuid.uuid4().hex, 16))


# Helper function for creating enum columns that store values as varchar instead of database enum
def EnumColumn(enum_class, **kwargs):
    """Create a String column for enum values to avoid database enum constraints"""
    # Remove enum-specific kwargs that don't apply to String columns
    kwargs.pop("name", None)

    # Determine the maximum length needed for enum values
    max_length = max(len(e.value) for e in enum_class) if enum_class and len(enum_class) > 0 else 50
    # Add some buffer for future enum values
    max_length = max(max_length + 20, 50)

    # Set default length if not specified
    kwargs.setdefault("length", max_length)

    return String(**kwargs)
