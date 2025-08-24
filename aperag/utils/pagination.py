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

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field
from sqlalchemy import Select, asc, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Generic pagination parameters"""

    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=100, description="Page size")


class SortParams(BaseModel):
    """Generic sorting parameters"""

    sort_by: Optional[str] = Field(None, description="Sort by field")
    sort_order: Optional[str] = Field("desc", description="Sort order")


class SearchParams(BaseModel):
    """Generic search parameters"""

    search: Optional[str] = Field(None, description="Search keyword")
    search_fields: Optional[List[str]] = Field(None, description="Search fields")


class ListParams(BaseModel):
    """Generic list query parameters"""

    pagination: PaginationParams = Field(default_factory=PaginationParams)
    sort: Optional[SortParams] = None
    search: Optional[SearchParams] = None
    filters: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response"""

    items: List[T]
    total: int = Field(description="Total count")
    page: int = Field(description="Current page")
    page_size: int = Field(description="Page size")
    total_pages: int = Field(description="Total pages")
    has_next: bool = Field(description="Has next page")
    has_prev: bool = Field(description="Has previous page")


class PaginationHelper:
    """Pagination helper class"""

    @staticmethod
    async def paginate_query(
        query: Select,
        session: AsyncSession,
        params: ListParams,
        sort_mapping: Optional[Dict[str, Any]] = None,
        search_fields: Optional[Dict[str, Any]] = None,
        default_sort: Optional[Any] = None,
    ) -> tuple[List, int]:
        """
        Apply pagination, sorting and search to SQLAlchemy query

        Args:
            query: SQLAlchemy query object
            session: Database session
            params: Query parameters
            sort_mapping: Sort field mapping {"field_name": Column}
            search_fields: Search field mapping {"field_name": Column}
            default_sort: Default sort field

        Returns:
            tuple: (items, total_count)
        """
        # Apply search filtering
        if params.search and params.search.search and search_fields:
            search_conditions = []
            search_term = f"%{params.search.search}%"

            # If specific search fields are provided, search only in those fields
            if params.search.search_fields:
                for field_name in params.search.search_fields:
                    if field_name in search_fields:
                        search_conditions.append(search_fields[field_name].ilike(search_term))
            else:
                # Otherwise search in all searchable fields
                for field in search_fields.values():
                    search_conditions.append(field.ilike(search_term))

            if search_conditions:
                query = query.where(or_(*search_conditions))

        # Apply custom filters
        if params.filters:
            for filter_key, filter_value in params.filters.items():
                if filter_value is not None:
                    # Can be extended with more complex filtering logic as needed
                    pass

        # Get total count (before applying sorting and pagination)
        from sqlalchemy import select

        count_query = select(func.count()).select_from(query.subquery())
        total = await session.scalar(count_query) or 0

        # Apply sorting
        if params.sort and params.sort.sort_by and sort_mapping:
            sort_field = sort_mapping.get(params.sort.sort_by)
            if sort_field is not None:
                if params.sort.sort_order == "asc":
                    query = query.order_by(asc(sort_field))
                else:
                    query = query.order_by(desc(sort_field))
        elif default_sort is not None:
            query = query.order_by(default_sort)

        # Apply pagination
        offset = (params.pagination.page - 1) * params.pagination.page_size
        query = query.offset(offset).limit(params.pagination.page_size)

        # Execute query
        result = await session.execute(query)
        items = result.scalars().all()

        return items, total

    @staticmethod
    def build_response(items: List[T], total: int, page: int, page_size: int) -> PaginatedResponse[T]:
        """Build paginated response"""
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1

        return PaginatedResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,  # Use requested page_size, not actual returned count
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
