from http import HTTPStatus

from asgiref.sync import async_to_sync
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management.base import BaseCommand, CommandError
from django.http import HttpRequest

from config import settings
from deeprag.utils.constant import KEY_USER_ID
from deeprag.views.main import create_document


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
        with open(path, 'rb') as fd:
            content = fd.read()
        file = SimpleUploadedFile(name=path, content=content, content_type=content_type)
        response = async_to_sync(create_document)(request, collection_id, [file])
        if int(response["code"]) != HTTPStatus.OK:
            raise CommandError('Failed to add document %s to collection: %s' % (path, response["message"]))
        elif len(response["data"]) == 0:
            raise CommandError('Failed to add document %s to collection, maybe unsupported suffix' % path)
        else:
            self.stdout.write(
                self.style.SUCCESS('Successfully added document %s with id %s to collection %s'
                                   % (path, response["data"][0]["id"], collection_id))
            )
