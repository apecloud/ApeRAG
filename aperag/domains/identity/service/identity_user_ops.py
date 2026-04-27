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

"""Identity-owned write facades for ``User`` fields that other domains
need to mutate.

G16 forbids consumer domains from importing the ``User`` ORM directly —
this module is the single narrow entry point identity-owned writes
travel through. Each facade is a thin wrapper over
``session.get(User, …)`` + attribute assignment so transaction semantics
stay aligned with the rest of the domain service layer (lesson 9a-sexdec
hierarchy-1 terminal state).
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.domains.identity.db.models import User


async def set_chat_collection(session: AsyncSession, user_id: str, collection_id: str) -> None:
    user = await session.get(User, user_id)
    if user is None:
        return
    user.chat_collection_id = collection_id
    session.add(user)
