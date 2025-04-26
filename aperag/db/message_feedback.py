from aperag.db.chat import Chat
from aperag.db.collection import Collection


from django.db import models


class MessageFeedbackStatus(models.TextChoices):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class MessageFeedback(models.Model):
    user = models.CharField(max_length=256)
    collection = models.ForeignKey(Collection, on_delete=models.CASCADE, null=True, blank=True)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE)
    message_id = models.CharField(max_length=256)
    upvote = models.IntegerField(default=0)
    downvote = models.IntegerField(default=0)
    relate_ids = models.TextField(null=True)
    question = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=MessageFeedbackStatus.choices, null=True)
    original_answer = models.TextField(null=True, blank=True)
    revised_answer = models.TextField(null=True, blank=True)
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('chat_id', 'message_id')