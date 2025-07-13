#!/bin/bash

set -o errexit
set -o nounset

exec watchfiles celery.__main__.main \
    --args \
    "-A config.celery flower --basic_auth=\"${CELERY_FLOWER_USER}:${CELERY_FLOWER_PASSWORD}\""
