"""
Prefect Configuration - Scheduled Flows and Deployments

This module contains configurations for Prefect scheduled flows and deployments.
It handles periodic tasks that were previously managed by Celery Beat.

Usage:
1. For development: Run flows locally using `.serve()` method
2. For production: Deploy using Prefect deployments

## Scheduling Reconcile Indexes

The index reconciliation flow needs to run periodically to maintain index consistency.
In production, this should be deployed as a scheduled deployment.

### Development Setup:
```python
from config.prefect_config import setup_development_schedules
setup_development_schedules()
```

### Production Deployment:
```bash
prefect deploy config/prefect_config.py:reconcile_indexes_deployment
```
"""

import logging
from datetime import timedelta

from prefect import serve
from prefect.client.schemas.schedules import IntervalSchedule
from config.prefect_tasks import reconcile_indexes_flow

logger = logging.getLogger(__name__)


def setup_development_schedules():
    """
    Setup scheduled flows for development environment
    
    This serves the reconcile_indexes_flow with a schedule.
    Call this function in development to start the scheduled flow server.
    """
    logger.info("Setting up Prefect development schedules")
    
    # Create interval schedule for reconcile indexes (every hour)
    schedule = IntervalSchedule(interval=timedelta(hours=1))
    
    # Serve the reconcile_indexes_flow with schedule
    reconcile_indexes_deployment = reconcile_indexes_flow.to_deployment(
        name="reconcile-indexes-dev",
        schedule=schedule,
        description="Periodic index reconciliation task (development)"
    )
    
    # Serve the deployment
    serve(reconcile_indexes_deployment)


def create_production_deployments():
    """
    Create production deployments for scheduled flows
    
    Returns list of deployments that can be deployed using Prefect CLI
    """
    # Create interval schedule for reconcile indexes (every hour)
    schedule = IntervalSchedule(interval=timedelta(hours=1))
    
    reconcile_indexes_deployment = reconcile_indexes_flow.to_deployment(
        name="reconcile-indexes-prod",
        schedule=schedule,
        description="Periodic index reconciliation task (production)",
        tags=["indexing", "reconciliation", "periodic"]
    )
    
    return [reconcile_indexes_deployment]


# Production deployment for CLI deployment
reconcile_indexes_deployment = reconcile_indexes_flow.to_deployment(
    name="reconcile-indexes",
    schedule=IntervalSchedule(interval=timedelta(hours=1)),
    description="Periodic index reconciliation task",
    tags=["indexing", "reconciliation", "periodic"]
)


if __name__ == "__main__":
    # For development - run this script to start scheduled flows
    setup_development_schedules() 