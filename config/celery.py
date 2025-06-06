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

import os
from celery import Celery
from aperag.config import settings

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
    include=['aperag.tasks.collection', 'aperag.tasks.index', 'aperag.tasks.crawl_web'],
)

# Set up task routes if local queue is specified
if settings.local_queue_name:
    app.conf.task_routes = {
        "aperag.tasks.index.add_index_for_local_document": {"queue": f"{settings.local_queue_name}"},
    }

if __name__ == "__main__":
    app.start()
