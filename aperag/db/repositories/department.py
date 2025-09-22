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

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from aperag.db.models import Department, DepartmentStatus
from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.utils.utils import utc_now


class AsyncDepartmentRepositoryMixin(AsyncRepositoryProtocol):
    async def get_department_by_id(self, department_id: str) -> Optional[Department]:
        """Get department by id"""
        async def _query(session):
            stmt = select(Department).where(Department.id == department_id)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def get_departments_by_tenant_id(self, tenant_id: str) -> List[Department]:
        """Get all departments by tenant id"""
        async def _query(session):
            stmt = select(Department).where(Department.tenant_id == tenant_id)
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def upsert_department(self, department_data: dict) -> Department:
        """Insert or update department data"""
        async def _operation(session):
            # Use PostgreSQL UPSERT (ON CONFLICT DO UPDATE)
            stmt = insert(Department).values(**department_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_={
                    'parent_id': stmt.excluded.parent_id,
                    'name': stmt.excluded.name,
                    'status': stmt.excluded.status,
                    'group_path': stmt.excluded.group_path,
                    'tenant_id': stmt.excluded.tenant_id,
                    'updated_at': utc_now(),
                }
            )
            stmt = stmt.returning(Department)
            result = await session.execute(stmt)
            await session.flush()
            return result.scalars().first()

        return await self.execute_with_transaction(_operation)

    async def bulk_upsert_departments(self, departments_data: List[dict]) -> List[Department]:
        """Bulk insert or update multiple departments"""
        async def _operation(session):
            results = []
            for dept_data in departments_data:
                stmt = insert(Department).values(**dept_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['id'],
                    set_={
                        'parent_id': stmt.excluded.parent_id,
                        'name': stmt.excluded.name,
                        'status': stmt.excluded.status,
                        'group_path': stmt.excluded.group_path,
                        'tenant_id': stmt.excluded.tenant_id,
                        'updated_at': utc_now(),
                    }
                )
                stmt = stmt.returning(Department)
                result = await session.execute(stmt)
                dept = result.scalars().first()
                if dept:
                    results.append(dept)
            
            await session.flush()
            return results

        return await self.execute_with_transaction(_operation)

    async def get_department_hierarchy(self, tenant_id: str, parent_id: str = "-1") -> List[Department]:
        """Get department hierarchy starting from parent_id"""
        async def _query(session):
            stmt = select(Department).where(
                Department.tenant_id == tenant_id,
                Department.parent_id == parent_id,
                Department.status == DepartmentStatus.ACTIVE
            ).order_by(Department.name)
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def get_department_children(self, department_id: str) -> List[Department]:
        """Get all child departments of a given department"""
        async def _query(session):
            stmt = select(Department).where(
                Department.parent_id == department_id,
                Department.status == DepartmentStatus.ACTIVE
            ).order_by(Department.name)
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)
