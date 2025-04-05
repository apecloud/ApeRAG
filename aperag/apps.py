import asyncio
import json

import requests
from asgiref.sync import sync_to_async
from django.apps import AppConfig


class ApeRAGConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "aperag"

    def ready(self):
        if asyncio.get_event_loop().is_running():
            asyncio.create_task(sync_to_async(get_ip_config)())


def get_ip_config():
    from django.db import transaction

    from aperag.db.models import Config

    try:
        public_ip = requests.get('https://ifconfig.me', timeout=5).text.strip()
    except Exception as e:
        print(e)
        return

    with transaction.atomic():
        Config.objects.get_or_create(key="public_ips", defaults={'value': '[]'})
        config = Config.objects.select_for_update().get(key="public_ips")

        public_ips = json.loads(config.value)
        if public_ip not in public_ips:
            public_ips.append(public_ip)
            config.value = json.dumps(public_ips)
            config.save()


class QuotaType:
    MAX_BOT_COUNT = 'max_bot_count'
    MAX_COLLECTION_COUNT = 'max_collection_count'
    MAX_DOCUMENT_COUNT = 'max_document_count'
    MAX_CONVERSATION_COUNT = 'max_conversation_count'
