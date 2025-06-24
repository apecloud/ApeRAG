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

KEY_USER_ID = "X-USER-ID"
KEY_BOT_ID = "X-BOT-ID"
KEY_CHAT_ID = "X-CHAT-ID"
KEY_WEBSOCKET_PROTOCOL = "Sec-Websocket-Protocol"
DOC_QA_REFERENCES = "|DOC_QA_REFERENCES|"
DOCUMENT_URLS = "|DOCUMENT_URLS|"


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


class QuotaType:
    MAX_BOT_COUNT = "max_bot_count"
    MAX_COLLECTION_COUNT = "max_collection_count"
    MAX_DOCUMENT_COUNT = "max_document_count"
    MAX_CONVERSATION_COUNT = "max_conversation_count"


class IndexAction:
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
