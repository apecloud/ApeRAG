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

"""Canonical ``indexing`` domain (bootstrap).

Phase 3 step 1 only lays down the Protocol surface the domain exposes
to its consumers (``knowledge_base`` above all). The domain body —
``manager.py`` / ``vector_index.py`` / ``fulltext_index.py`` /
``graph_index.py`` / ``summary_index.py`` / ``vision_index.py`` /
``document_parser.py`` / ``base.py`` — is moved here from
``aperag/index/`` in Phase 3 step 2 as an atomic ``git mv``. Until that
move lands, importing the domain's runtime members still goes through
``aperag.index.*``; only the Protocols in ``ports.py`` are resolved
through this package.
"""
