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

"""Legacy shim for ``aperag/views/bots_v2.py`` after Phase 5 step 5-S4g.

Handlers merged with ``aperag/views/chat.py`` into the single
``aperag.domains.conversation.api.routes`` module. The v2 bot-scope
router (``bots_router``) is re-exported here as ``router`` so
pre-migration callers that did ``from aperag.views.bots_v2 import
router`` keep resolving the same object. Phase 6 cleanup removes
this shim once every caller has migrated to the domain path.
"""

from aperag.domains.conversation.api.routes import bots_router as router  # noqa: F401

__all__ = ["router"]
