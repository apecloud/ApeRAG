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

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from aperag.config import settings

# Import Neo4j sync configuration signal handlers
try:
    from aperag.db.neo4j_sync_manager import setup_worker_neo4j, cleanup_worker_neo4j
    neo4j_available = True
except ImportError:
    setup_worker_neo4j = None
    cleanup_worker_neo4j = None
    neo4j_available = False

# Create celery app instance
app = Celery("aperag")

# Configure celery
app.conf.update(
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
    include=['aperag.tasks.task_collection', 'aperag.tasks.task_document'],
)

app.conf.beat_schedule = {
    'reconcile-indexes': {
        'task': 'aperag.tasks.task_document.reconcile_indexes_task',
        'schedule': 5.0,
    },
}

# Set up task routes if local queue is specified
if settings.local_queue_name:
    app.conf.task_routes = {
        "aperag.tasks.index.add_index_for_local_document": {"queue": f"{settings.local_queue_name}"},
    }

# Connect Neo4j worker lifecycle signals using correct syntax
if neo4j_available and setup_worker_neo4j is not None:
    worker_process_init.connect(setup_worker_neo4j)

if neo4j_available and cleanup_worker_neo4j is not None:
    worker_process_shutdown.connect(cleanup_worker_neo4j)

@worker_process_init.connect
def setup_worker(**kwargs):
    """Additional worker setup if needed"""
    pass

@worker_process_shutdown.connect
def shutdown_worker(**kwargs):
    """Additional worker cleanup if needed"""
    pass

if __name__ == "__main__":
    app.start()
