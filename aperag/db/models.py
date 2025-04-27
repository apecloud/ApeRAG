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

import random
import uuid

from django.db import models
from django.db.models import IntegerField
from django.db.models.functions import Cast
from django.utils import timezone
from django.contrib.auth.models import AbstractUser
from datetime import timedelta

from config import settings


def random_id():
    """Generate a random ID string"""
    return ''.join(random.sample(uuid.uuid4().hex, 16))


def upload_document_path(document, filename):
    user = document.user.replace("|", "-")
    return "documents/user-{0}/{1}/{2}".format(
        user, document.collection_id, filename
    )


class Collection(models.Model):
    class Status(models.TextChoices):
        INACTIVE = "INACTIVE"
        ACTIVE = "ACTIVE"
        DELETED = "DELETED"
        QUESTION_PENDING = "QUESTION_PENDING"

    class SyncStatus(models.TextChoices):
        RUNNING = "RUNNING"
        CANCELED = "CANCELED"
        COMPLETED = "COMPLETED"

    class Type(models.TextChoices):
        DOCUMENT = "document"

    @staticmethod
    def generate_id():
        """Generate a random ID for collection"""
        return "col" + random_id()

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    title = models.CharField(max_length=256)
    description = models.TextField(null=True, blank=True)
    user = models.CharField(max_length=256)
    status = models.CharField(max_length=16, choices=Status.choices)
    type = models.CharField(max_length=16, choices=Type.choices)
    config = models.TextField()
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    async def bots(self, only_ids=False):
        """Get all active bots related to this collection"""
        from aperag.db import models as db_models
        result = []
        async for rel in db_models.BotCollectionRelation.objects.filter(collection_id=self.id, gmt_deleted__isnull=True).all():
            if only_ids:
                result.append(rel.bot_id)
            else:
                bot = await db_models.Bot.objects.aget(id=rel.bot_id)
                result.append(bot)
        return result


class Document(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING"
        RUNNING = "RUNNING"
        COMPLETE = "COMPLETE"
        FAILED = "FAILED"
        DELETING = "DELETING"
        DELETED = "DELETED"
        WARNING = "WARNING"

    class ProtectAction(models.TextChoices):
        WARNING_NOT_STORED = "nostore"
        REPLACE_WORDS = "replace"

    @staticmethod
    def generate_id():
        """Generate a random ID for document"""
        return "doc" + random_id()

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    name = models.CharField(max_length=1024)
    user = models.CharField(max_length=256)
    config = models.TextField(null=True)
    collection_id = models.CharField(max_length=24, null=True)
    status = models.CharField(max_length=16, choices=Status.choices)
    size = models.BigIntegerField()
    file = models.FileField(upload_to=upload_document_path, max_length=1024)
    relate_ids = models.TextField()
    metadata = models.TextField(default="{}")
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)
    sensitive_info = models.JSONField(default=list)

    class Meta:
        unique_together = ('collection_id', 'name')

    async def get_collection(self):
        """Get the associated collection object"""
        try:
            return await Collection.objects.aget(id=self.collection_id)
        except Collection.DoesNotExist:
            return None

    @property
    async def collection(self):
        """Property to maintain backwards compatibility"""
        return await self.get_collection()

    @collection.setter 
    async def collection(self, collection):
        """Setter to maintain backwards compatibility"""
        if isinstance(collection, Collection):
            self.collection_id = collection.id
        elif isinstance(collection, str):
            self.collection_id = collection


class Bot(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "ACTIVE"
        DELETED = "DELETED"

    class Type(models.TextChoices):
        KNOWLEDGE = "knowledge"
        COMMON = "common"

    @staticmethod
    def generate_id():
        """Generate a random ID for bot"""
        return "bot" + random_id()

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    user = models.CharField(max_length=256)
    title = models.CharField(max_length=256)
    type = models.CharField(max_length=16, choices=Type.choices, default=Type.KNOWLEDGE)
    description = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices)
    config = models.TextField()
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    async def collections(self, only_ids=False):
        """Get all active collections related to this bot"""
        from aperag.db import models as db_models
        result = []
        async for rel in db_models.BotCollectionRelation.objects.filter(bot_id=self.id, gmt_deleted__isnull=True).all():
            if only_ids:
                result.append(rel.collection_id)
            else:
                collection = await db_models.Collection.objects.aget(id=rel.collection_id)
                result.append(collection)
        return result


class BotCollectionRelation(models.Model):
    bot_id = models.CharField(max_length=24)
    collection_id = models.CharField(max_length=24)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["bot_id", "collection_id"],
                condition=models.Q(gmt_deleted__isnull=True),
                name="unique_active_bot_collection"
            )
        ]


class Config(models.Model):
    key = models.CharField(max_length=256, unique=True)
    value = models.TextField()
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)


class UserQuota(models.Model):
    user = models.CharField(max_length=256)
    key = models.CharField(max_length=256)
    value = models.PositiveIntegerField(default=0)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)


class Chat(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "ACTIVE"
        DELETED = "DELETED"

    class PeerType(models.TextChoices):
        SYSTEM = "system"
        FEISHU = "feishu"
        WEIXIN = "weixin"
        WEIXIN_OFFICIAL = "weixin_official"
        WEB = "web"
        DINGTALK = "dingtalk"

    @staticmethod
    def generate_id():
        """Generate a random ID for chat"""
        return "chat" + random_id()

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    user = models.CharField(max_length=256)
    peer_type = models.CharField(max_length=16, default=PeerType.SYSTEM, choices=PeerType.choices)
    peer_id = models.CharField(max_length=256, null=True)
    status = models.CharField(max_length=16, choices=Status.choices)
    bot_id = models.CharField(max_length=24)
    title = models.TextField()
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('bot_id', 'peer_type', 'peer_id')

    async def get_bot(self):
        """Get the associated bot object"""
        try:
            return await Bot.objects.aget(id=self.bot_id)
        except Bot.DoesNotExist:
            return None

    @property
    async def bot(self):
        """Property to maintain backwards compatibility"""
        return await self.get_bot()

    @bot.setter
    async def bot(self, bot):
        """Setter to maintain backwards compatibility"""
        if isinstance(bot, Bot):
            self.bot_id = bot.id
        elif isinstance(bot, str):
            self.bot_id = bot


class MessageFeedback(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING"
        RUNNING = "RUNNING"
        COMPLETE = "COMPLETE"
        FAILED = "FAILED"

    user = models.CharField(max_length=256)
    collection_id = models.CharField(max_length=24, null=True, blank=True)
    chat_id = models.CharField(max_length=24)
    message_id = models.CharField(max_length=256)
    upvote = models.IntegerField(default=0)
    downvote = models.IntegerField(default=0)
    relate_ids = models.TextField(null=True)
    question = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, null=True)
    original_answer = models.TextField(null=True, blank=True)
    revised_answer = models.TextField(null=True, blank=True)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('chat_id', 'message_id')

    async def get_collection(self):
        """Get the associated collection object"""
        try:
            return await Collection.objects.aget(id=self.collection_id)
        except Collection.DoesNotExist:
            return None

    @property
    async def collection(self):
        """Property to maintain backwards compatibility"""
        return await self.get_collection()

    @collection.setter
    async def collection(self, collection):
        """Setter to maintain backwards compatibility"""
        if isinstance(collection, Collection):
            self.collection_id = collection.id
        elif isinstance(collection, str):
            self.collection_id = collection

    async def get_chat(self):
        """Get the associated chat object"""
        try:
            return await Chat.objects.aget(id=self.chat_id)
        except Chat.DoesNotExist:
            return None

    @property
    async def chat(self):
        """Property to maintain backwards compatibility"""
        return await self.get_chat()

    @chat.setter
    async def chat(self, chat):
        """Setter to maintain backwards compatibility"""
        if isinstance(chat, Chat):
            self.chat_id = chat.id
        elif isinstance(chat, str):
            self.chat_id = chat


class Question(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "ACTIVE"
        WARNING = "WARNING"
        DELETED = "DELETED"
        PENDING = "PENDING"

    @staticmethod
    def generate_id():
        """Generate a random ID for question"""
        return "que" + random_id()

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    user = models.CharField(max_length=256)
    collection_id = models.CharField(max_length=24)
    documents = models.ManyToManyField(Document, blank=True)
    question = models.TextField()
    answer = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, null=True)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)
    relate_id = models.CharField(null=True, max_length=256)

    async def get_collection(self):
        """Get the associated collection object"""
        try:
            return await Collection.objects.aget(id=self.collection_id)
        except Collection.DoesNotExist:
            return None

    @property
    async def collection(self):
        """Property to maintain backwards compatibility"""
        return await self.get_collection()

    @collection.setter
    async def collection(self, collection):
        """Setter to maintain backwards compatibility"""
        if isinstance(collection, Collection):
            self.collection_id = collection.id
        elif isinstance(collection, str):
            self.collection_id = collection


class ApiKey(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "ACTIVE"
        DELETED = "DELETED"

    @staticmethod
    def generate_id():
        """Generate a random ID for API key"""
        return ''.join(random.sample(uuid.uuid4().hex, 12))

    @staticmethod
    def generate_key():
        """Generate a random API key with sk- prefix"""
        return f"sk-{uuid.uuid4().hex}"

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    key = models.CharField(max_length=40, default=generate_key.__func__, editable=False)
    user = models.CharField(max_length=256)
    description = models.CharField(max_length=256, blank=True, null=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.ACTIVE)
    last_used_at = models.DateTimeField(null=True, blank=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"ApiKeyToken(id={self.id}, user={self.user}, description={self.description})"

    def update_last_used(self):
        """Update the last used timestamp"""
        self.last_used_at = timezone.now()
        self.save()


class CollectionSyncHistory(models.Model):
    @staticmethod
    def generate_id():
        """Generate a random ID for collection sync history"""
        return "colhist" + random_id()

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    user = models.CharField(max_length=256)
    collection_id = models.CharField(max_length=24)
    total_documents = models.PositiveIntegerField(default=0)
    new_documents = models.PositiveIntegerField(default=0)
    deleted_documents = models.PositiveIntegerField(default=0)
    modified_documents = models.PositiveIntegerField(default=0)
    processing_documents = models.PositiveIntegerField(default=0)
    pending_documents = models.PositiveIntegerField(default=0)
    failed_documents = models.PositiveIntegerField(default=0)
    successful_documents = models.PositiveIntegerField(default=0)
    total_documents_to_sync = models.PositiveIntegerField(default=0)
    execution_time = models.DurationField(null=True)
    start_time = models.DateTimeField()
    task_context = models.JSONField(default=dict)
    status = models.CharField(max_length=16, choices=Collection.SyncStatus.choices, default=Collection.SyncStatus.RUNNING)
    gmt_created = models.DateTimeField(auto_now_add=True, null=True)
    gmt_updated = models.DateTimeField(auto_now=True, null=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    def update_execution_time(self):
        self.refresh_from_db()
        self.execution_time = timezone.now() - self.start_time
        self.save()

    async def collection(self):
        """Get the associated collection object"""
        try:
            return await Collection.objects.aget(id=self.collection_id)
        except Collection.DoesNotExist:
            return None

    def view(self):
        return {
            "id": str(self.id),
            "user": str(self.user),
            "total_documents": self.total_documents,
            "new_documents": self.new_documents,
            "deleted_documents": self.deleted_documents,
            "pending_documents": self.pending_documents,
            "processing_documents": self.processing_documents,
            "modified_documents": self.modified_documents,
            "failed_documents": self.failed_documents,
            "successful_documents": self.successful_documents,
            "total_documents_to_sync": self.total_documents_to_sync,
            "start_time": self.start_time,
            "execution_time": self.execution_time,
            "status": self.status,
        }


class ModelServiceProvider(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "ACTIVE"
        INACTIVE = "INACTIVE"
        DELETED = "DELETED"

    @staticmethod
    def generate_id():
        """Generate a random ID for model service provider"""
        return "int" + random_id()

    name = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    user = models.CharField(max_length=256)
    status = models.CharField(max_length=16, choices=Status.choices)
    base_url = models.CharField(max_length=256, blank=True, null=True)
    api_key = models.CharField(max_length=256)
    extra = models.TextField(null=True)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"ModelServiceProvider(name={self.name}, user={self.user}, status={self.status}, base_url={self.base_url}, api_key={self.api_key}, extra={self.extra})"


class Role(models.TextChoices):
    ADMIN = "admin"
    RW = "rw"
    RO = "ro"
    

class User(AbstractUser):
    """Custom user model that extends AbstractUser"""
    email = models.EmailField(unique=True, blank=True, null=True)
    role = models.CharField(max_length=16, choices=Role.choices, default=Role.RO)

    class Meta:
        app_label = 'aperag'  # Set app_label to 'aperag' since we're using AUTH_USER_MODEL = 'aperag.User'


class Invitation(models.Model):
    """Invitation model for user registration"""
    @staticmethod
    def generate_id():
        """Generate a random ID for invitation"""
        return "invite" + random_id()

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    email = models.EmailField()
    token = models.CharField(max_length=64, unique=True)
    created_by = models.CharField(max_length=256)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_used = models.BooleanField(default=False)
    used_at = models.DateTimeField(null=True, blank=True)
    role = models.CharField(max_length=16, choices=Role.choices, default=Role.RO)
    
    class Meta:
        app_label = 'aperag'
        
    async def asave(self, *args, **kwargs):
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(days=7)
        await super().asave(*args, **kwargs)
        
    def is_valid(self):
        return not self.is_used and self.expires_at > timezone.now()
        
    async def use(self):
        self.is_used = True
        self.used_at = timezone.now()
        await self.asave()
