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
        user, document.collection.id, filename
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

    def view(self, bot_ids=None):
        if not bot_ids:
            bot_ids = []
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "type": self.type,
            "bot_ids": bot_ids,
            "system": self.user == settings.ADMIN_USER,
            "config": self.config,
            "created": self.gmt_created.isoformat(),
            "updated": self.gmt_updated.isoformat(),
        }


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
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE)
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
        unique_together = ('collection', 'name')

    def collection_id(self):
        if self.collection:
            return Cast(self.collection, IntegerField())
        else:
            return None

    collection_id.short_description = 'Collection ID'
    collection_id.admin_order_field = 'collection'


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
    collections = models.ManyToManyField(Collection)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)


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
    bot = models.ForeignKey(Bot, on_delete=models.CASCADE)
    title = models.TextField()
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('bot', 'peer_type', 'peer_id')


class MessageFeedback(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING"
        RUNNING = "RUNNING"
        COMPLETE = "COMPLETE"
        FAILED = "FAILED"

    user = models.CharField(max_length=256)
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, null=True, blank=True)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE)
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
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE)
    documents = models.ManyToManyField(Document, blank=True)
    question = models.TextField()
    answer = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, null=True)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)
    relate_id = models.CharField(null=True, max_length=256)


class ApiKeyToken(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "ACTIVE"
        DELETED = "DELETED"

    @staticmethod
    def generate_id():
        """Generate a random ID for API key"""
        return ''.join(random.sample(uuid.uuid4().hex, 12))

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    key = models.CharField(max_length=40, editable=False)
    user = models.CharField(max_length=256)
    status = models.CharField(max_length=16, choices=Status.choices, null=True)
    count_times = models.IntegerField(blank=True, null=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)


class CollectionSyncHistory(models.Model):
    @staticmethod
    def generate_id():
        """Generate a random ID for collection sync history"""
        return "colhist" + random_id()

    id = models.CharField(primary_key=True, default=generate_id.__func__, editable=False, max_length=24)
    user = models.CharField(max_length=256)
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE)
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
