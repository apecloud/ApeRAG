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

from logging.config import dictConfig

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from aperag.config import settings

# Create celery app instance
app = Celery("aperag")

# Configure celery
app.conf.update(
    task_acks_late=True,
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_send_task_events=settings.celery_worker_send_task_events,
    task_send_sent_event=settings.celery_task_send_sent_event,
    task_track_started=settings.celery_task_track_started,
    # Auto-discover tasks in the aperag.tasks package
    include=['config.celery_tasks'],
    # Enable detailed logging for celery workers
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(name)s - %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(name)s - %(message)s',
    worker_hijack_root_logger=False,  # Don't hijack root logger
)

app.conf.beat_schedule = {
    'reconcile-indexes': {
        'task': 'config.celery_tasks.reconcile_indexes_task',
        'schedule': 3600.0,  # Increased from 600s (10min) to 3600s (60min) since we now trigger on document operations
    },
}

# Set up task routes if local queue is specified
if settings.local_queue_name:
    app.conf.task_routes = {
        "aperag.tasks.index.add_index_for_local_document": {"queue": f"{settings.local_queue_name}"},
    }


# Simple logging configuration for Celery workers
CELERY_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'console': {
            'format': '[%(asctime)s: %(levelname)s/%(processName)s] %(name)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'console',
            'level': 'INFO',
        }
    },
    'loggers': {
        'LiteLLM': {
            'level': 'WARNING',
            'handlers': ['console'],
            'propagate': False,
        }
    }
}

@worker_process_init.connect
def setup_worker(**kwargs):
    """Setup logging and other worker initialization"""
    # Configure logging for this worker process
    dictConfig(CELERY_LOGGING_CONFIG)

@worker_process_shutdown.connect
def shutdown_worker(**kwargs):
    """Additional worker cleanup if needed"""
    pass

if __name__ == "__main__":
    app.start()
