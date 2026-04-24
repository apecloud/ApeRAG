# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Internal pipeline for graphindex v2. NOT a public import surface.

Business code must import from ``aperag.domains.knowledge_graph.graphindex`` (the facade),
never directly from ``aperag.domains.knowledge_graph.graphindex.engine``.
"""

from aperag.domains.knowledge_graph.graphindex.engine.indexer import index_document

__all__ = ["index_document"]
