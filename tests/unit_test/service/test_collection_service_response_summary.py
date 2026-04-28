# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""``CollectionService.build_collection_response`` must include
``summary`` so the FE settings page can render the auto-generated
``Collection.summary`` after a regen call.

Regression caught by @earayu2 msg=e4120886 (2026-04-29): the toast said
"摘要重新生成成功" but the textarea remained empty. Root cause was the
response builder dropping ``instance.summary`` even though the
``Collection`` Pydantic schema declares it. This test pins the field.
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from aperag.domains.knowledge_base.service.collection_service import CollectionService


@pytest.mark.asyncio
async def test_build_collection_response_includes_summary_field():
    instance = SimpleNamespace(
        id="col-1",
        title="t",
        description="d",
        summary="A long auto-generated summary " * 10,
        type="document",
        status="ACTIVE",
        config=json.dumps({"language": "zh-CN"}),
        gmt_created=datetime(2026, 4, 29, 0, 0, 0),
        gmt_updated=datetime(2026, 4, 29, 0, 0, 0),
    )
    response = await CollectionService().build_collection_response(instance)

    assert response.summary == instance.summary, (
        "build_collection_response must echo Collection.summary so the FE settings "
        "page can render the auto-generated summary after a regen call"
    )


@pytest.mark.asyncio
async def test_build_collection_response_summary_none_when_db_column_null():
    """A collection that has not been summarised yet (column is NULL)
    must still build cleanly and surface ``summary=None`` to the FE."""
    instance = SimpleNamespace(
        id="col-2",
        title="t",
        description=None,
        summary=None,
        type="document",
        status="ACTIVE",
        config=json.dumps({"language": "en-US"}),
        gmt_created=datetime(2026, 4, 29, 0, 0, 0),
        gmt_updated=datetime(2026, 4, 29, 0, 0, 0),
    )
    response = await CollectionService().build_collection_response(instance)
    assert response.summary is None
