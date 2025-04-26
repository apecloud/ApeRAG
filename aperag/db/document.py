from aperag.db.collection import Collection


from django.db import models
from django.db.models import IntegerField
from django.db.models.functions import Cast

from aperag.db.utils import random_id


class DocumentStatus(models.TextChoices):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    DELETING = "DELETING"
    DELETED = "DELETED"
    WARNING = "WARNING"


def upload_document_path(document, filename):
    user = document.user.replace("|", "-")
    return "documents/user-{0}/{1}/{2}".format(
        user, document.collection.id, filename
    )


def doc_pk():
    return "doc" + random_id()


class Document(models.Model):
    id = models.CharField(primary_key=True, default=doc_pk, editable=False, max_length=24)
    name = models.CharField(max_length=1024)
    user = models.CharField(max_length=256)
    config = models.TextField(null=True)
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE)
    status = models.CharField(max_length=16, choices=DocumentStatus.choices)
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

    def view(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "status": self.status,
            "config": self.metadata,
            "size": self.size,
            "created": self.gmt_created.isoformat(),
            "updated": self.gmt_updated.isoformat(),
            "sensitive_info": self.sensitive_info,
        }

    # def collection_id(self):
    #     if self.collection:
    #         matches = re.findall(r'\d+', str(self.collection))
    #         return matches[0] if matches else '-'
    #     else:
    #         return '-'

    def collection_id(self):
        if self.collection:
            return Cast(self.collection, IntegerField())
        else:
            return None

    collection_id.short_description = 'Collection ID'
    collection_id.admin_order_field = 'collection'