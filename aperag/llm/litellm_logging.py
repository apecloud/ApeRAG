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


def setup_litellm_logging():
    import logging

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False
    logging.info("LiteLLM logging is set to WARNING level and propagation is disabled.")

    import litellm

    litellm.suppress_debug_info = True
    logging.info("LiteLLM debug info suppression is enabled.")

    # Filter Pydantic serialization warnings globally
    import warnings

    warnings.filterwarnings(
        "ignore", category=UserWarning, module="pydantic.*", message=".*Pydantic serializer warnings.*"
    )

    # Also filter the specific warnings we're seeing
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Expected 9 fields but got.*")

    warnings.filterwarnings("ignore", category=UserWarning, message=".*Expected.*StreamingChoices.*but got.*Choices.*")
