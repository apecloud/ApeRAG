from aperag import settings
from aperag.store.collection import Collection


from django.db import models

from aperag.store.utils import random_id


class BotStatus(models.TextChoices):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class BotType(models.TextChoices):
    KNOWLEDGE = "knowledge"
    COMMON = "common"


def bot_pk():
    return "bot" + random_id()


class Bot(models.Model):
    id = models.CharField(primary_key=True, default=bot_pk, editable=False, max_length=24)
    user = models.CharField(max_length=256)
    title = models.CharField(max_length=256)
    type = models.CharField(max_length=16, choices=BotType.choices, default=BotType.KNOWLEDGE)
    description = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=BotStatus.choices)
    config = models.TextField()
    collections = models.ManyToManyField(Collection)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    def view(self, collections=None):
        if collections is None:
            collections = []
        return {
            "id": str(self.id),
            "title": self.title,
            "type": self.type,
            "description": self.description,
            "config": self.config,
            "system": self.user == settings.ADMIN_USER,
            "collections": collections,
            "created": self.gmt_created.isoformat(),
            "updated": self.gmt_updated.isoformat(),
        }


class BotIntegrationStatus(models.TextChoices):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DELETED = "DELETED"


class BotIntegrationType(models.TextChoices):
    SYSTEM = "system"
    FEISHU = "feishu"
    WEB = "web"
    WEIXN = "weixin"
    WEIXIN_OFFICIAL = "weixin_official"