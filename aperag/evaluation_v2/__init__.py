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

"""Legacy shim package — canonical home is
``aperag.domains.evaluation`` after Phase 5 step 5-S6.

The submodules at ``aperag/evaluation_v2/{constants,judges,schemas,
services,tasks,worker}.py`` are retained as thin re-export shims so
pre-migration callers (views/evaluation_v2.py shim, Celery task
imports, unit tests that patch ``aperag.evaluation_v2.worker.*``)
keep resolving the same objects without a rename sweep. Phase 6
cleanup deletes the whole package + its submodule shims in one
sweep.
"""

EVALUATION_V2_SCHEMA_VERSION = "evaluation-v2.0"
