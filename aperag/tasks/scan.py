import json
import logging

from celery import Task
from django_celery_beat.models import CrontabSchedule, PeriodicTask

from aperag.db.models import Collection, CollectionStatus

logger = logging.getLogger(__name__)


class CustomScanTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        collection_id = args[0]
        collection = Collection.objects.get(id=collection_id)
        collection.status = CollectionStatus.ACTIVE
        collection.save()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        # todo: when scan fail we should do something
        raise exc
        # collection_id = args[0]
        # collection = Collection.objects.get(id=collection_id)
        # collection.status = CollectionStatus.INACTIVE
        # collection.save()


async def update_sync_documents_cron_job(collection_id):
    collection = await Collection.objects.aget(id=collection_id)
    task = await get_schedule_task(collection_id)
    config = json.loads(collection.config)
    if "crontab" not in config or not config["crontab"] or not config["crontab"].get("enabled", False):
        if await task.acount():
            await task.aupdate(enabled=False)
            await task.adelete()
        return

    crontab, _ = await CrontabSchedule.objects.aupdate_or_create(
        minute=config["crontab"]["minute"],
        hour=config["crontab"]["hour"],
        day_of_week=config["crontab"]["day_of_week"],
        day_of_month=config["crontab"]["day_of_month"],
        # timezone="Etc/GMT-" + config["crontab"]["UTC"]
    )
    if await task.acount():
        await task.aupdate(crontab=crontab)
    else:
        await PeriodicTask.objects.acreate(
            name="collection-" + str(collection.id) + "-sync-documents",
            kwargs=json.dumps({"collection_id": str(collection.id)}),
            task="aperag.tasks.sync_documents_task.sync_documents",
            crontab=crontab
        )
    logger.info(f"update sync documents cronjob for collection{collection_id}")


async def delete_sync_documents_cron_job(collection_id):
    task = await get_schedule_task(collection_id)
    if await task.acount():
        await task.aupdate(enabled=False)
        await task.adelete()
        logger.info(f"delete sync documents cronjob for collection{collection_id}")


async def get_schedule_task(collection_id):
    return PeriodicTask.objects.filter(name="collection-" + str(collection_id) + "-sync-documents")


