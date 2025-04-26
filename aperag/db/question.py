from aperag.db.collection import Collection
from aperag.db.document import Document


from django.db import models

from aperag.db.utils import random_id


class QuestionStatus(models.TextChoices):
    ACTIVE = "ACTIVE"
    WARNING = "WARNING"
    DELETED = "DELETED"
    PENDING = "PENDING"


def que_pk():
    return "que" + random_id()


class Question(models.Model):
    id = models.CharField(primary_key=True, default=que_pk, editable=False, max_length=24)
    user = models.CharField(max_length=256)
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE)
    documents = models.ManyToManyField(Document, blank=True)
    question = models.TextField()
    answer = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=QuestionStatus.choices, null=True)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)
    relate_id = models.CharField(null=True, max_length=256)

    def view(self, relate_documents=None):
        if not relate_documents:
            relate_documents = []
        return {
            "id": str(self.id),
            "status": self.status,
            "question": self.question,
            "answer": self.answer,
            "relate_documents": relate_documents,
            "created": self.gmt_created.isoformat(),
            "updated": self.gmt_updated.isoformat(),
        }