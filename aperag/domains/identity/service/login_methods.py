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

from aperag.config import settings


def is_google_oauth_enabled() -> bool:
    return bool(settings.google_oauth_client_id and settings.google_oauth_client_secret)


def is_github_oauth_enabled() -> bool:
    return bool(settings.github_oauth_client_id and settings.github_oauth_client_secret)


def get_available_login_methods() -> list[str]:
    methods = ["local"]
    if is_google_oauth_enabled():
        methods.append("google")
    if is_github_oauth_enabled():
        methods.append("github")
    return methods
