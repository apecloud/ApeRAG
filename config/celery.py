import os

from celery import Celery

# set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("aperag")

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

from config.settings import LOCAL_QUEUE_NAME  # noqa: E402

if LOCAL_QUEUE_NAME != "":
    app.conf.task_routes = {
        "aperag.tasks.index.add_index_for_local_document": {"queue": f"{LOCAL_QUEUE_NAME}"},
    }

if __name__ == "__main__":
    app.start()
