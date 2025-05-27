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

from abc import ABC, abstractmethod
from typing import List

from langchain.schema import AIMessage, BaseMessage, HumanMessage


class BaseChatMessageHistory(ABC):
    """Abstract base class for storing chat message history.

    See `ChatMessageHistory` for default implementation.

    Example:
        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_message(self, message: BaseMessage) -> None:
                   messages = self.messages.append(_message_to_dict(message))
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    async def add_user_message(self, message: str) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
        """
        await self.add_message(HumanMessage(content=message))

    async def add_ai_message(self, message: str) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
        """
        await self.add_message(AIMessage(content=message))

    @abstractmethod
    async def add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store.
        """
        raise NotImplementedError()

    @abstractmethod
    async def clear(self) -> None:
        """Remove all messages from the store"""

    @property
    async def messages(self) -> List[BaseMessage]:
        """Retrieve all messages from the store.

        Returns:
            A list of BaseMessage objects.
        """
        raise NotImplementedError()
