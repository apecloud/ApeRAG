from django.db import models


class Config(models.Model):
    key = models.CharField(max_length=256, unique=True)
    value = models.TextField()
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)