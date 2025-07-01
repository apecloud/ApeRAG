#!/bin/bash

set -o errexit
set -o nounset

export LOCAL_QUEUE_NAME=${NODE_IP}
if [ -z "${LOCAL_QUEUE_NAME}" ]; then
    export LOCAL_QUEUE_NAME="localhost"
fi

exec celery -A config.celery worker -l INFO --concurrency=16 -Q ${LOCAL_QUEUE_NAME},celery --pool=threads
