# KubeChat

KubeChat is a chat tool for KubeBlocks, it is used for Text2SQL, Text2CD, Text2CV, Text2Cluster, Chatbot and
troubleshooting. KubeChat adopts the open-source LLM for local serving, so it can promise 100% privacy.

# Development Guide

You should install Python 3.11 first.

* install poetry

```bash
pip3.11 install poetry
```

* install dependencies

```bash
poetry install
```

* prepare configs

```bash
cp envs/.env.template .env
```

* prepare postgres/redis/qdrant/elasticsearch

```bash
make run-db
```

* run the django service

```bash
poetry shell

make run-backend
```

* run the celery service

```bash
poetry shell

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python celery -A config.celery worker -l INFO --pool gevent
```

* run the frontend

There are two options to run the frontend, one is to run the frontend from source directly, the other is to run the frontend in docker.

Option 1:

```bash

git clone https://github.com/apecloud/KubeChat-FrontEnd.git

cd KubeChat-FrontEnd

echo 'API_ENDPOINT=http://127.0.0.1:8000' >> .env

yarn dev

```

Option 2 [Not Ready]:

```bash
docker run --rm -p 8001:8001 -e "API_ENDPOINT=http://127.0.0.1:8000" apecloud/kubechat-console:latest
```


then open the kubechat frontend console: http://localhost:8001
