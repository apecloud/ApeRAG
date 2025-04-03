from http import HTTPStatus

from asgiref.sync import async_to_sync
from django.core.management.base import BaseCommand, CommandError
from django.http import HttpRequest

from config import settings
from deeprag.utils.constant import KEY_USER_ID
from deeprag.views.main import delete_document


class Command(BaseCommand):
    help = "delete document to the default collection"

    def add_arguments(self, parser):
        parser.add_argument("--cid", type=str, required=True, help="collection id")
        parser.add_argument("--did", type=str, required=True, help="document id")

    def handle(self, *args, **options):
        collection_id = options["cid"]
        document_id = options["did"]

        request = HttpRequest()
        request.method = "POST"
        request.META = {
            KEY_USER_ID: settings.ADMIN_USER,
        }
        response = async_to_sync(delete_document)(request, collection_id, document_id)
        if int(response["code"]) != HTTPStatus.OK:
            raise CommandError('Failed to delete document %s in collection %s: %s'
                               % (document_id, collection_id, response["message"]))
        else:
            self.stdout.write(
                self.style.SUCCESS('Successfully deleted document %s in collection %s' % (document_id, collection_id))
            )
