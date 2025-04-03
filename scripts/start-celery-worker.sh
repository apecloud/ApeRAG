#!/bin/bash

set -o errexit
set -o nounset

export LOCAL_QUEUE_NAME=${NODE_IP}
if [ -z "${LOCAL_QUEUE_NAME}" ]; then
    export LOCAL_QUEUE_NAME="localhost"
fi

# https://github.com/apecloud/ApeRAG/issues/331
# https://stackoverflow.com/questions/63645357/using-pytorch-with-celery
#case "${EMBEDDING_DEVICE}" in
#    "cpu")
#        echo "Using CPU to embedding"
#        pool="prefork"
#        ;;
#    "cuda")
#        echo "Using CUDA to embedding"
#        pool="solo"
#        ;;
#    *)
#        echo "Unknown EMBEDDING_DEVICE: ${EMBEDDING_DEVICE}"
#        exit 1
#        ;;
#esac
pool=solo

exec celery -A config.celery worker -l INFO --concurrency 1 -Q ${LOCAL_QUEUE_NAME},celery --pool=${pool}
