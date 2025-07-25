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

"""Chat history message types and utilities."""

from .message import (
    StoredChatMessage,
    StoredChatMessagePart,
    create_assistant_message,
    create_user_message,
    group_messages_by_message_id,
    group_parts_by_message_id,
    message_to_storage_dict,
    messages_to_frontend_format,
    messages_to_openai_format,
    storage_dict_to_message,
)

__all__ = [
    # Core message classes
    "StoredChatMessagePart",
    "StoredChatMessage",
    # Helper functions for creating messages
    "create_user_message",
    "create_assistant_message",
    # Conversion functions
    "message_to_storage_dict",
    "storage_dict_to_message",
    "group_messages_by_message_id",
    "group_parts_by_message_id",
    "messages_to_frontend_format",
    "messages_to_openai_format",
]
