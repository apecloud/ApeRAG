"""
Celery Beat configuration for K8s-inspired reconciliation tasks

This module defines the periodic tasks that should be run by Celery Beat
to maintain the document index reconciliation system.
"""


# Celery Beat schedule configuration
CELERY_BEAT_SCHEDULE = {
    # Run reconciliation every 30 seconds
    'reconcile-document-indexes': {
        'task': 'aperag.tasks.index_tasks.reconcile_indexes_task',
        'schedule': 30.0,  # Every 30 seconds
        'options': {
            'expires': 25,  # Task expires after 25 seconds to avoid overlap
        }
    },
}

# Timezone for the scheduler
CELERY_TIMEZONE = 'UTC' 