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
from celery.signals import before_task_publish, task_postrun, task_prerun, worker_process_init, worker_process_shutdown

from aperag.config import settings
from aperag.observability import build_observability_config, configure_logging, configure_process_observability
from aperag.observability.context import bind_observability_context, reset_observability_context
from aperag.observability.tracing import attach_context_from_carrier, detach_context, inject_carrier, start_span

observability_config = build_observability_config(settings)
configure_logging(observability_config)
configure_process_observability(observability_config)

# Create celery app instance
app = Celery("aperag")

# Configure celery
app.conf.update(
    task_acks_late=True,
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_send_task_events=settings.celery_worker_send_task_events,
    task_send_sent_event=settings.celery_task_send_sent_event,
    task_track_started=settings.celery_task_track_started,
    # Auto-discover tasks in the aperag.tasks package
    include=[
        "aperag.domains.indexing.tasks",
        "aperag.domains.knowledge_base.tasks",
        "aperag.domains.knowledge_graph.tasks",
        "aperag.domains.evaluation.tasks",
    ],
    # Enable detailed logging for celery workers - let our custom config handle formatting
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(name)s - %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(name)s - %(message)s",
    # Let our custom logging configuration handle the root logger
    worker_hijack_root_logger=True,
)

app.conf.beat_schedule = {
    "reconcile-indexes": {
        "task": "config.celery_tasks.reconcile_indexes_task",
        "schedule": 300.0,  # Run every 5 minutes
    },
    "reconcile-collection-summaries": {
        "task": "config.celery_tasks.reconcile_collection_summaries_task",
        "schedule": 60.0,
    },
    "collection-gc": {
        "task": "config.celery_tasks.cleanup_expired_documents_task",
        "schedule": 600.0,
    },
}


@worker_process_init.connect
def setup_worker(**kwargs):
    """Setup logging and other worker initialization"""
    configure_logging(observability_config)
    configure_process_observability(observability_config)
    # Celery tasks create isolated event loops (`asyncio.run()` / manual loop wrappers).
    # LiteLLM's async callback worker keeps a process-global asyncio.Queue, which can become
    # bound to the wrong loop and crash the worker process.
    from aperag.llm.litellm_logging import disable_litellm_async_logging_callbacks

    disable_litellm_async_logging_callbacks()


@before_task_publish.connect
def inject_trace_context(headers=None, **kwargs):
    if headers is not None:
        inject_carrier(headers)


@task_prerun.connect
def start_task_observability(task=None, task_id=None, **kwargs):
    if task is None:
        return
    headers = getattr(getattr(task, "request", None), "headers", None) or {}
    token = attach_context_from_carrier(headers)
    context_tokens = bind_observability_context(task_id=task_id, operation=getattr(task, "name", None))
    span_cm = start_span(
        "celery.task.run",
        tracer_name="aperag.celery",
        **{
            "aperag.task.id": task_id,
            "aperag.task.name": getattr(task, "name", None),
        },
    )
    span_cm.__enter__()
    task.request._aperag_observability_token = token
    task.request._aperag_observability_context_tokens = context_tokens
    task.request._aperag_observability_span_cm = span_cm


@task_postrun.connect
def finish_task_observability(task=None, state=None, **kwargs):
    if task is None:
        return
    request = getattr(task, "request", None)
    span_cm = getattr(request, "_aperag_observability_span_cm", None)
    if span_cm is not None:
        span_cm.__exit__(None, None, None)
    reset_observability_context(getattr(request, "_aperag_observability_context_tokens", None))
    detach_context(getattr(request, "_aperag_observability_token", None))


@worker_process_shutdown.connect
def shutdown_worker(**kwargs):
    """Additional worker cleanup if needed"""
    pass


if __name__ == "__main__":
    app.start()
