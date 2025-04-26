# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    from aperag.db.config import Config

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
