from http import HTTPStatus

from asgiref.sync import async_to_sync
from django.core.management.base import BaseCommand, CommandError
from django.http import HttpRequest

from config import settings
from deeprag.utils.constant import KEY_USER_ID
from deeprag.views.main import delete_collection


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
            raise CommandError('Failed to delete collection %s: %s' % (collection_id, response["message"]))
        else:
            self.stdout.write(
                self.style.SUCCESS('Successfully deleted collection %s' % response["data"]["id"])
            )
