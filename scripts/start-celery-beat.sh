#!/bin/bash

set -o errexit
set -o nounset


rm -f './celerybeat.pid'
exec watchfiles celery.__main__.main --args '-A config.celery beat -l INFO'
