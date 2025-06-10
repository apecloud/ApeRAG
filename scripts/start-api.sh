#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset


python3 manage.py migrate

if [ -f "static/web/index.html" ]; then
  cp static/web/index.html aperag/templates/404.html
fi

# uvicorn supports multiple workers
exec uvicorn config.asgi:application --host 0.0.0.0 --log-config scripts/uvicorn-log-config.yaml

#exec daphne config.asgi:application -b 0.0.0.0