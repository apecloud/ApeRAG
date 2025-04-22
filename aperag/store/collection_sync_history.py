from aperag.store.collection import Collection, CollectionSyncStatus


from django.db import models
from django.utils import timezone

from aperag.store.utils import random_id


def collection_history_pk():
    return "colhist" + random_id()


class CollectionSyncHistory(models.Model):
    id = models.CharField(primary_key=True, default=collection_history_pk, editable=False, max_length=24)
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
    status = models.CharField(max_length=16, choices=CollectionSyncStatus.choices, default=CollectionSyncStatus.RUNNING)
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