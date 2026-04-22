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

import pytest

from aperag.objectstore import base as objectstore_base
from aperag.objectstore.local import AsyncLocal, Local, LocalConfig
from aperag.objectstore.s3 import S3, S3Config


@pytest.fixture(autouse=True)
def reset_objectstore_singletons():
    old_sync = objectstore_base._SYNC_STORE
    old_async = objectstore_base._ASYNC_STORE
    objectstore_base._SYNC_STORE = None
    objectstore_base._ASYNC_STORE = None
    try:
        yield
    finally:
        objectstore_base._SYNC_STORE = old_sync
        objectstore_base._ASYNC_STORE = old_async


def test_get_object_store_returns_local_singleton(monkeypatch, tmp_path):
    monkeypatch.setattr(objectstore_base.settings, "object_store_type", "local")
    monkeypatch.setattr(
        objectstore_base.settings,
        "object_store_local_config",
        LocalConfig(root_dir=str(tmp_path / "sync-store")),
    )
    monkeypatch.setattr(objectstore_base.settings, "object_store_s3_config", None)

    first = objectstore_base.get_object_store()
    second = objectstore_base.get_object_store()

    assert isinstance(first, Local)
    assert first is second


@pytest.mark.asyncio
async def test_get_async_object_store_returns_local_singleton(monkeypatch, tmp_path):
    monkeypatch.setattr(objectstore_base.settings, "object_store_type", "local")
    monkeypatch.setattr(
        objectstore_base.settings,
        "object_store_local_config",
        LocalConfig(root_dir=str(tmp_path / "async-store")),
    )
    monkeypatch.setattr(objectstore_base.settings, "object_store_s3_config", None)

    first = objectstore_base.get_async_object_store()
    second = objectstore_base.get_async_object_store()

    assert isinstance(first, AsyncLocal)
    assert first is second


def test_get_object_store_returns_s3_singleton(monkeypatch):
    monkeypatch.setattr(objectstore_base.settings, "object_store_type", "s3")
    monkeypatch.setattr(objectstore_base.settings, "object_store_local_config", None)
    monkeypatch.setattr(
        objectstore_base.settings,
        "object_store_s3_config",
        S3Config(
            endpoint="http://localhost:9000",
            access_key="ak",
            secret_key="sk",
            bucket="bucket",
            region="us-east-1",
            use_path_style=True,
        ),
    )

    first = objectstore_base.get_object_store()
    second = objectstore_base.get_object_store()

    assert isinstance(first, S3)
    assert first is second


@pytest.mark.asyncio
async def test_get_async_object_store_requires_s3_config(monkeypatch):
    monkeypatch.setattr(objectstore_base.settings, "object_store_type", "s3")
    monkeypatch.setattr(objectstore_base.settings, "object_store_local_config", None)
    monkeypatch.setattr(objectstore_base.settings, "object_store_s3_config", None)

    with pytest.raises(RuntimeError, match="OBJECT_STORE_S3_"):
        objectstore_base.get_async_object_store()
