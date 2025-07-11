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

import json

from aperag.vectorstore.connector import VectorStoreConnectorAdaptor
from aperag.config import settings


def get_vector_db_connector(collection: str) -> VectorStoreConnectorAdaptor:
    # todo: specify the collection for different user
    # one person one collection
    ctx = json.loads(settings.vector_db_context)
    ctx["collection"] = collection
    return VectorStoreConnectorAdaptor(settings.vector_db_type, ctx=ctx)


# vector_db_connector = get_local_vector_db_connector(VECTOR_DB_TYPE)
