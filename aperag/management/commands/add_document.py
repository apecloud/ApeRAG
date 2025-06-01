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
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management.base import BaseCommand, CommandError
from django.http import HttpRequest

from aperag.utils.constant import KEY_USER_ID
from aperag.views.main import create_documents
from config import settings


class Command(BaseCommand):
    help = "add document to the default collection"

    def add_arguments(self, parser):
        parser.add_argument("--cid", type=str, required=True, help="collection id")
        parser.add_argument("--path", type=str, default="", required=True, help="document path")
        parser.add_argument("--type", type=str, default="text/plain", help="document content type")

    def handle(self, *args, **options):
        collection_id = options["cid"]
        path = options["path"]
        content_type = options["type"]

        request = HttpRequest()
        request.method = "POST"
        request.META = {
            KEY_USER_ID: settings.ADMIN_USER,
        }
        with open(path, "rb") as fd:
            content = fd.read()
        file = SimpleUploadedFile(name=path, content=content, content_type=content_type)
        response = async_to_sync(create_documents)(request, collection_id, [file])
        if int(response["code"]) != HTTPStatus.OK:
            raise CommandError("Failed to add document %s to collection: %s" % (path, response["message"]))
        elif len(response["data"]) == 0:
            raise CommandError("Failed to add document %s to collection, maybe unsupported suffix" % path)
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    "Successfully added document %s with id %s to collection %s"
                    % (path, response["data"][0]["id"], collection_id)
                )
            )
