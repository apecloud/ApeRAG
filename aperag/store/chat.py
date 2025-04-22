from aperag.store.bot import Bot


from django.db import models

from aperag.store.utils import random_id


class ChatStatus(models.TextChoices):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class ChatPeer(models.TextChoices):
    SYSTEM = "system"
    FEISHU = "feishu"
    WEIXIN = "weixin"
    WEIXIN_OFFICIAL = "weixin_official"
    WEB = "web"
    DINGTALK = "dingtalk"


def chat_pk():
    return "chat" + random_id()


class Chat(models.Model):
    id = models.CharField(primary_key=True, default=chat_pk, editable=False, max_length=24)
    user = models.CharField(max_length=256)
    peer_type = models.CharField(max_length=16, default=ChatPeer.SYSTEM, choices=ChatPeer.choices)
    peer_id = models.CharField(max_length=256, null=True)
    status = models.CharField(max_length=16, choices=ChatStatus.choices)
    bot = models.ForeignKey(Bot, on_delete=models.CASCADE)
    summary = models.TextField()
    gmt_created = models.DateTimeField(auto_now_add=True)
    gmt_updated = models.DateTimeField(auto_now=True)
    gmt_deleted = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ('bot', 'peer_type', 'peer_id')

    def view(self, bot_id, messages=None):
        if messages is None:
            messages = []
        return {
            "id": str(self.id),
            "summary": self.summary,
            "bot_id": bot_id,
            "history": messages,
            "peer_type": self.peer_type,
            "peer_id": self.peer_id or "",
            "created": self.gmt_created.isoformat(),
            "updated": self.gmt_updated.isoformat(),
        }