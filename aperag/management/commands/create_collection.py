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

import json
from http import HTTPStatus

from asgiref.sync import async_to_sync
from django.core.management.base import BaseCommand, CommandError
from django.http import HttpRequest

from config import settings
from aperag.utils.constant import KEY_USER_ID
from aperag.views.main import create_collection
from aperag.schema.view_models import CollectionCreate


class Command(BaseCommand):
    help = "init the default collections"

    def add_arguments(self, parser):
        parser.add_argument("--title", type=str, required=True, help="collection title")
        parser.add_argument("--description", type=str, default="", help="collection description")

    def handle(self, *args, **options):
        title = options["title"]
        description = options["description"]

        request = HttpRequest()
        request.method = "POST"
        request.META = {
            KEY_USER_ID: settings.ADMIN_USER,
        }
        config = {
            "source": "system"
        }
        data = CollectionCreate(type="document", title=title, description=description, config=json.dumps(config))
        response = async_to_sync(create_collection)(request, data)
        if int(response["code"]) != HTTPStatus.OK:
            raise CommandError('Failed to create collection: %s' % response.get("message", "unknown error"))
        else:
            self.stdout.write(
                self.style.SUCCESS('Successfully created collection %s' % response["data"]["id"])
            )
