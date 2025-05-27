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

from http import HTTPStatus

from asgiref.sync import async_to_sync
from django.core.management.base import BaseCommand, CommandError
from django.http import HttpRequest

from aperag.utils.constant import KEY_USER_ID
from aperag.views.main import delete_collection
from config import settings


class Command(BaseCommand):
    help = "delete the default collection"

    def add_arguments(self, parser):
        parser.add_argument("--cid", type=str, required=True, help="collection id")

    def handle(self, *args, **options):
        collection_id = options["cid"]

        request = HttpRequest()
        request.method = "DELETE"
        request.META = {
            KEY_USER_ID: settings.ADMIN_USER,
        }
        response = async_to_sync(delete_collection)(request, collection_id)
        if int(response["code"]) != HTTPStatus.OK:
            raise CommandError("Failed to delete collection %s: %s" % (collection_id, response["message"]))
        else:
            self.stdout.write(self.style.SUCCESS("Successfully deleted collection %s" % response["data"]["id"]))
