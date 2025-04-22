from django.db import models


class UserQuota(models.Model):
    user = models.CharField(max_length=256)
    key = models.CharField(max_length=256)
    value = models.PositiveIntegerField(default=0)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)