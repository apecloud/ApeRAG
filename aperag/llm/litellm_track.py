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
from datetime import datetime

import litellm
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


class MyCustomHandler(CustomLogger):
    def __init__(self):
        self.prefix = "LiteLLM"

    def _format_time(self, dt: datetime) -> str:
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def log_pre_api_call(self, model, messages, kwargs):
        pass

    # async func won't trigger this callback, it's a bug. https://github.com/BerriAI/litellm/issues/8842
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            error_str = str(response_obj) if response_obj else "Unknown error"
            logger.error(f"{self.prefix} Error: {error_str}", exc_info=True)
        except Exception as e:
            logger.error(f"{self.prefix} Error in log_failure_event", exc_info=e)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        if kwargs["call_type"] in ["completion", "acompletion"]:
            # self.log_post_api_call_for_completion(kwargs, response_obj, start_time, end_time)
            pass
        else:
            # self.log_post_api_call_for_completion(kwargs, response_obj, start_time, end_time)
            pass

    def log_post_api_call_for_completion(self, kwargs, response_obj, start_time, end_time):
        try:
            end_time = end_time or datetime.now()
            latency = (end_time - start_time).total_seconds() if start_time and end_time else 0.0
            cost = kwargs.get("response_cost", 0) or 0

            logger.info(f"{self.prefix} Model: {kwargs.get('model')}")
            logger.info(f"{self.prefix} Provider: {kwargs.get('custom_llm_provider', 'unknown')}")
            logger.info(f"{self.prefix} base_url: {kwargs.get('base_url', 'unknown')}")
            logger.info(f"{self.prefix} Start Time: {self._format_time(start_time)}")
            logger.info(f"{self.prefix} End Time: {self._format_time(end_time)}")
            logger.info(f"{self.prefix} Latency: {latency:.2f} seconds")
            logger.info(f"{self.prefix} Cost: ${cost:.6f}")
        except Exception as e:
            logger.error(f"{self.prefix} Error in log_success_event", exc_info=e)


def register_custom_llm_track():
    litellm.callbacks = [MyCustomHandler()]


def register_opik_llm_track():
    from litellm.integrations.opik.opik import OpikLogger

    from aperag.config import settings

    if settings.opik_api_key and settings.opik_workspace:
        litellm.callbacks = [OpikLogger()]
