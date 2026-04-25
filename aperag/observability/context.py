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

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

request_id_var: ContextVar[Optional[str]] = ContextVar("aperag_request_id", default=None)
task_id_var: ContextVar[Optional[str]] = ContextVar("aperag_task_id", default=None)
domain_var: ContextVar[Optional[str]] = ContextVar("aperag_domain", default=None)
operation_var: ContextVar[Optional[str]] = ContextVar("aperag_operation", default=None)


@dataclass(frozen=True)
class ObservabilityContext:
    request_id: Optional[str] = None
    task_id: Optional[str] = None
    domain: Optional[str] = None
    operation: Optional[str] = None


def get_request_id() -> Optional[str]:
    return request_id_var.get()


def get_task_id() -> Optional[str]:
    return task_id_var.get()


def get_domain() -> Optional[str]:
    return domain_var.get()


def get_operation() -> Optional[str]:
    return operation_var.get()


def get_observability_context() -> ObservabilityContext:
    return ObservabilityContext(
        request_id=get_request_id(),
        task_id=get_task_id(),
        domain=get_domain(),
        operation=get_operation(),
    )


def bind_observability_context(
    *,
    request_id: Optional[str] = None,
    task_id: Optional[str] = None,
    domain: Optional[str] = None,
    operation: Optional[str] = None,
):
    tokens = []
    if request_id is not None:
        tokens.append((request_id_var, request_id_var.set(request_id)))
    if task_id is not None:
        tokens.append((task_id_var, task_id_var.set(task_id)))
    if domain is not None:
        tokens.append((domain_var, domain_var.set(domain)))
    if operation is not None:
        tokens.append((operation_var, operation_var.set(operation)))
    return tokens


def reset_observability_context(tokens) -> None:
    for var, token in reversed(tokens or []):
        var.reset(token)


def clear_observability_context():
    return bind_observability_context(request_id="", task_id="", domain="", operation="")


@contextmanager
def operation_context(
    *,
    request_id: Optional[str] = None,
    task_id: Optional[str] = None,
    domain: Optional[str] = None,
    operation: Optional[str] = None,
) -> Iterator[None]:
    tokens = bind_observability_context(
        request_id=request_id,
        task_id=task_id,
        domain=domain,
        operation=operation,
    )

    try:
        yield
    finally:
        reset_observability_context(tokens)
