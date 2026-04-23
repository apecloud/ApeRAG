# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Storage backends for graphindex v2."""

from aperag.domains.knowledge_graph.graphindex.storage.base import GraphStore
from aperag.domains.knowledge_graph.graphindex.storage.connector import GraphStoreAdaptor
from aperag.domains.knowledge_graph.graphindex.storage.postgres import PostgresGraphStore

__all__ = ["GraphStore", "GraphStoreAdaptor", "PostgresGraphStore"]
