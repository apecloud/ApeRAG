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

"""
Celery Beat schedule configuration for declarative index management
"""

from celery.schedules import crontab

# Beat schedule for declarative index processing
CELERY_BEAT_SCHEDULE = {
    # Process pending document indexes every 30 seconds
    'process-pending-indexes': {
        'task': 'aperag.tasks.celery_tasks.process_pending_document_indexes',
        'schedule': 30.0,  # Run every 30 seconds
        'args': (20,),  # Process up to 20 indexes at once
        'options': {
            'expires': 300,  # Task expires after 5 minutes
        }
    },
    
    # Retry failed indexes every 5 minutes
    'retry-failed-indexes': {
        'task': 'aperag.tasks.celery_tasks.retry_failed_document_indexes',
        'schedule': crontab(minute='*/5'),  # Run every 5 minutes
        'args': (10,),  # Retry up to 10 failed indexes at once
        'options': {
            'expires': 600,  # Task expires after 10 minutes
        }
    },
    
    # Cleanup completed index records older than 7 days - run daily at 2 AM
    'cleanup-old-index-records': {
        'task': 'aperag.tasks.celery_tasks.cleanup_old_index_records',
        'schedule': crontab(hour=2, minute=0),  # Run daily at 2 AM
        'options': {
            'expires': 3600,  # Task expires after 1 hour
        }
    },
    
    # Cleanup expired index locks every 10 minutes
    'cleanup-expired-locks': {
        'task': 'aperag.tasks.celery_tasks.cleanup_expired_index_locks',
        'schedule': crontab(minute='*/10'),  # Run every 10 minutes
        'options': {
            'expires': 300,  # Task expires after 5 minutes
        }
    },
}

# Additional beat configuration
CELERY_BEAT_SCHEDULER = 'django_celery_beat.schedulers:DatabaseScheduler'
CELERY_TIMEZONE = 'UTC' 