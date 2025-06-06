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
from datetime import timedelta

from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


def random_id():
    """Generate a random ID string"""
    return "".join(random.sample(uuid.uuid4().hex, 16))


class Collection(models.Model):
    class Status(models.TextChoices):
        INACTIVE = "INACTIVE"
        ACTIVE = "ACTIVE"
        DELETED = "DELETED"

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
        async for rel in db_models.BotCollectionRelation.objects.filter(
            collection_id=self.id, gmt_deleted__isnull=True
        ).all():
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

    class IndexStatus(models.TextChoices):
        PENDING = "PENDING"
        RUNNING = "RUNNING"
        COMPLETE = "COMPLETE"
        FAILED = "FAILED"
        SKIPPED = "SKIPPED"  # When certain functionality is not enabled

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

    # Independent index status fields for different index types
    vector_index_status = models.CharField(max_length=16, choices=IndexStatus.choices, default=IndexStatus.PENDING)
    fulltext_index_status = models.CharField(max_length=16, choices=IndexStatus.choices, default=IndexStatus.PENDING)
    graph_index_status = models.CharField(max_length=16, choices=IndexStatus.choices, default=IndexStatus.PENDING)

    size = models.BigIntegerField()
    object_path = models.CharField(max_length=1024, null=True)
    relate_ids = models.TextField()
    metadata = models.TextField(default="{}")
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ("collection_id", "name")

    def object_store_base_path(self) -> str:
        user = self.user.replace("|", "-")
        return "user-{0}/{1}/{2}".format(user, self.collection_id, self.id)

    def get_overall_status(self):
        """Calculate overall status based on individual index statuses"""
        index_statuses = [self.vector_index_status, self.fulltext_index_status, self.graph_index_status]

        # If any index failed, overall status is failed
        if any(status == self.IndexStatus.FAILED for status in index_statuses):
            return self.Status.FAILED
        # If any index is running, overall status is running
        elif any(status == self.IndexStatus.RUNNING for status in index_statuses):
            return self.Status.RUNNING
        # If all indexes are complete or skipped, overall status is complete
        elif all(status in [self.IndexStatus.COMPLETE, self.IndexStatus.SKIPPED] for status in index_statuses):
            return self.Status.COMPLETE
        # Otherwise, status is pending
        else:
            return self.Status.PENDING

    def update_overall_status(self):
        """Update overall status field"""
        self.status = self.get_overall_status()

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
                name="unique_active_bot_collection",
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
        unique_together = ("bot_id", "peer_type", "peer_id")

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

    class Type(models.TextChoices):
        GOOD = "good"
        BAD = "bad"

    class Tag(models.TextChoices):
        HARMFUL = "Harmful"
        UNSAFE = "Unsafe"
        FAKE = "Fake"
        UNHELPFUL = "Unhelpful"
        OTHER = "Other"

    user = models.CharField(max_length=256)
    collection_id = models.CharField(max_length=24, null=True, blank=True)
    chat_id = models.CharField(max_length=24)
    message_id = models.CharField(max_length=256)
    type = models.CharField(max_length=16, choices=Type.choices, null=True)
    tag = models.CharField(max_length=16, choices=Tag.choices, null=True)
    message = models.TextField(null=True, blank=True)
    question = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, null=True)
    original_answer = models.TextField(null=True, blank=True)
    revised_answer = models.TextField(null=True, blank=True)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ("chat_id", "message_id")

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


class ApiKey(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "ACTIVE"
        DELETED = "DELETED"

    @staticmethod
    def generate_id():
        """Generate a random ID for API key"""
        return "".join(random.sample(uuid.uuid4().hex, 12))

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

    async def update_last_used(self):
        """Update the last used timestamp"""
        self.last_used_at = timezone.now()
        await self.asave()


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
    status = models.CharField(
        max_length=16, choices=Collection.SyncStatus.choices, default=Collection.SyncStatus.RUNNING
    )
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

    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=256)
    user = models.CharField(max_length=256)
    status = models.CharField(max_length=16, choices=Status.choices)
    api_key = models.CharField(max_length=256)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [("name", "user")]

    def __str__(self):
        return f"ModelServiceProvider(name={self.name}, user={self.user}, status={self.status}, api_key={self.api_key})"


class Role(models.TextChoices):
    ADMIN = "admin"
    RW = "rw"
    RO = "ro"


class User(AbstractUser):
    """Custom user model that extends AbstractUser"""

    email = models.EmailField(unique=True, blank=True, null=True)
    role = models.CharField(max_length=16, choices=Role.choices, default=Role.RO)

    class Meta:
        app_label = "aperag"  # Set app_label to 'aperag' since we're using AUTH_USER_MODEL = 'aperag.User'


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
        app_label = "aperag"

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


class SearchTestHistory(models.Model):
    @staticmethod
    def generate_id():
        """Generate a random ID for search test history"""
        return "sth" + random_id()

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    user = models.CharField(max_length=256)
    collection_id = models.CharField(max_length=24, null=True)
    query = models.TextField()
    vector_search = models.JSONField(null=True, blank=True)
    fulltext_search = models.JSONField(null=True, blank=True)
    graph_search = models.JSONField(null=True, blank=True)
    items = models.JSONField(default=list)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)


class LLMProvider(models.Model):
    """LLM Provider configuration model

    This model stores the provider-level configuration that was previously
    stored in model_configs.json file. Each provider has basic information
    and dialect configurations for different API types.
    """

    name = models.CharField(max_length=128, primary_key=True, help_text="Unique provider name identifier")
    label = models.CharField(max_length=256, help_text="Human-readable provider display name")
    completion_dialect = models.CharField(max_length=64, help_text="API dialect for completion/chat APIs")
    embedding_dialect = models.CharField(max_length=64, help_text="API dialect for embedding APIs")
    rerank_dialect = models.CharField(max_length=64, help_text="API dialect for rerank APIs")
    allow_custom_base_url = models.BooleanField(default=False, help_text="Whether custom base URLs are allowed")
    base_url = models.URLField(max_length=512, help_text="Default API base URL for this provider")
    extra = models.TextField(null=True, help_text="Additional configuration data in JSON format")
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "aperag_llm_provider"
        app_label = "aperag"

    def __str__(self):
        return f"LLMProvider(name={self.name}, label={self.label})"


class LLMProviderModel(models.Model):
    """LLM Provider Model configuration

    This model stores individual model configurations for each provider.
    Each model belongs to a provider and has a specific API type (completion, embedding, rerank).
    """

    class APIType(models.TextChoices):
        COMPLETION = "completion", "Completion"
        EMBEDDING = "embedding", "Embedding"
        RERANK = "rerank", "Rerank"

    provider_name = models.CharField(max_length=128, help_text="Reference to LLMProvider.name")
    api = models.CharField(max_length=16, choices=APIType.choices, help_text="API type for this model")
    model = models.CharField(max_length=256, help_text="Model name/identifier")
    custom_llm_provider = models.CharField(max_length=128, help_text="Custom LLM provider implementation")
    max_tokens = models.PositiveIntegerField(null=True, blank=True, help_text="Maximum tokens for this model")
    tags = models.JSONField(
        default=list, blank=True, help_text="Tags for model categorization, e.g. ['free', 'recommend']"
    )
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "aperag_llm_provider_models"
        app_label = "aperag"
        unique_together = [("provider_name", "api", "model")]
        indexes = [
            models.Index(fields=["provider_name"]),
            models.Index(fields=["api"]),
            models.Index(fields=["provider_name", "api"]),
        ]

    def __str__(self):
        return f"LLMProviderModel(provider={self.provider_name}, api={self.api}, model={self.model})"

    async def get_provider(self):
        """Get the associated provider object"""
        try:
            return await LLMProvider.objects.aget(name=self.provider_name)
        except LLMProvider.DoesNotExist:
            return None

    @property
    async def provider(self):
        """Property to get the provider object"""
        return await self.get_provider()

    def has_tag(self, tag: str) -> bool:
        """Check if model has a specific tag"""
        return tag in (self.tags or [])

    def add_tag(self, tag: str) -> bool:
        """Add a tag to model. Returns True if tag was added, False if already exists"""
        if self.tags is None:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
            return True
        return False

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from model. Returns True if tag was removed, False if not found"""
        if self.tags and tag in self.tags:
            self.tags.remove(tag)
            return True
        return False

    def get_tags(self) -> list:
        """Get all tags for this model"""
        return self.tags or []
