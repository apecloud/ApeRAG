

import random
import uuid
from django.db import models


class ApiKeyStatus(models.TextChoices):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


def app_id():
    return ''.join(random.sample(uuid.uuid4().hex, 12))


class ApiKeyToken(models.Model):
    id = models.CharField(primary_key=True, default=app_id, editable=False, max_length=24)
    key = models.CharField(max_length=40, editable=False)
    user = models.CharField(max_length=256)
    status = models.CharField(max_length=16, choices=ApiKeyStatus.choices, null=True)
    count_times = models.IntegerField(blank=True, null=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    def view(self):
        return {
            "id": str(self.id),
            "key": self.key,
        }