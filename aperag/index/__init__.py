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

"""
K8s-Inspired Document Index Management System

This module provides a simple, reliable system for managing document indexes
using a declarative approach inspired by Kubernetes resource management.

Key components:
- ReconciliationController: Ensures actual state matches desired state
- IndexSpecManager: API for managing index specifications
- SimpleIndexService: High-level service API for applications

The system uses two main tables:
- DocumentIndexSpec: Declares desired state (present/absent)
- DocumentIndexStatus: Tracks actual state (absent/creating/present/deleting/failed)

A reconciliation controller runs periodically to ensure consistency.
"""

from .reconciliation_controller import (
    IndexSpecManager,
    ReconciliationController,
    index_spec_manager,
    reconciliation_controller,
)

__all__ = [
    'ReconciliationController',
    'IndexSpecManager', 
    'reconciliation_controller',
    'index_spec_manager'
] 