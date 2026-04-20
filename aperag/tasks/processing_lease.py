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

import logging
import os
import threading
import uuid
from datetime import timedelta
from typing import Callable, Optional

from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)

DEFAULT_PROCESSING_LEASE_TTL_SECONDS = int(os.getenv("APERAG_PROCESSING_LEASE_TTL_SECONDS", "900"))
DEFAULT_PROCESSING_LEASE_RENEW_INTERVAL_SECONDS = int(os.getenv("APERAG_PROCESSING_LEASE_RENEW_INTERVAL_SECONDS", "60"))


def generate_processing_token() -> str:
    return uuid.uuid4().hex


def build_lease_expires_at(ttl_seconds: int = DEFAULT_PROCESSING_LEASE_TTL_SECONDS):
    return utc_now() + timedelta(seconds=ttl_seconds)


class ProcessingLeaseRenewer:
    """Background helper that periodically renews the current processing lease."""

    def __init__(
        self,
        renew_fn: Callable[[], bool],
        *,
        interval_seconds: int = DEFAULT_PROCESSING_LEASE_RENEW_INTERVAL_SECONDS,
        description: str,
    ):
        self._renew_fn = renew_fn
        self._interval_seconds = max(interval_seconds, 1)
        self._description = description
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.ownership_lost = False

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"lease-renewer:{self._description}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 1)

    def _run(self):
        while not self._stop_event.wait(self._interval_seconds):
            try:
                renewed = self._renew_fn()
            except Exception:
                logger.exception("Processing lease renewer failed for %s", self._description)
                continue

            if renewed:
                continue

            self.ownership_lost = True
            logger.warning("Processing lease ownership lost for %s", self._description)
            self._stop_event.set()
            return
