#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset


alembic -c aperag/alembic.ini upgrade head

exec uvicorn aperag.app:app --host 0.0.0.0 --log-config scripts/uvicorn-log-config.yaml