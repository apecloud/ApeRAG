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

from typing import TypeVar, Generic, Optional, List, Dict, Any, Callable
from pydantic import BaseModel, Field
from sqlalchemy import Select, func, desc, asc, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

T = TypeVar('T')

class PaginationParams(BaseModel):
    """通用分页参数"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=10, ge=1, le=100, description="每页大小")

class SortParams(BaseModel):
    """通用排序参数"""
    sort_by: Optional[str] = Field(None, description="排序字段")
    sort_order: Optional[str] = Field('desc', description="排序方向")

class SearchParams(BaseModel):
    """通用搜索参数"""
    search: Optional[str] = Field(None, description="搜索关键词")
    search_fields: Optional[List[str]] = Field(None, description="搜索字段")

class ListParams(BaseModel):
    """通用列表查询参数"""
    pagination: PaginationParams = Field(default_factory=PaginationParams)
    sort: Optional[SortParams] = None
    search: Optional[SearchParams] = None
    filters: Optional[Dict[str, Any]] = None

class PaginatedResponse(BaseModel, Generic[T]):
    """通用分页响应"""
    items: List[T]
    total: int = Field(description="总数")
    page: int = Field(description="当前页")
    page_size: int = Field(description="每页大小")
    total_pages: int = Field(description="总页数")
    has_next: bool = Field(description="是否有下一页")
    has_prev: bool = Field(description="是否有上一页")

class PaginationHelper:
    """分页助手类"""
    
    @staticmethod
    async def paginate_query(
        query: Select,
        session: AsyncSession,
        params: ListParams,
        sort_mapping: Optional[Dict[str, Any]] = None,
        search_fields: Optional[Dict[str, Any]] = None,
        default_sort: Optional[Any] = None
    ) -> tuple[List, int]:
        """
        对SQLAlchemy查询应用分页、排序和搜索
        
        Args:
            query: SQLAlchemy查询对象
            session: 数据库会话
            params: 查询参数
            sort_mapping: 排序字段映射 {"field_name": Column}
            search_fields: 搜索字段映射 {"field_name": Column}
            default_sort: 默认排序字段
            
        Returns:
            tuple: (items, total_count)
        """
        # 1. 应用搜索过滤
        if params.search and params.search.search and search_fields:
            search_conditions = []
            search_term = f"%{params.search.search}%"
            
            # 如果指定了搜索字段，只在这些字段中搜索
            if params.search.search_fields:
                for field_name in params.search.search_fields:
                    if field_name in search_fields:
                        search_conditions.append(search_fields[field_name].ilike(search_term))
            else:
                # 否则在所有可搜索字段中搜索
                for field in search_fields.values():
                    search_conditions.append(field.ilike(search_term))
            
            if search_conditions:
                query = query.where(or_(*search_conditions))
        
        # 2. 应用自定义过滤器
        if params.filters:
            for filter_key, filter_value in params.filters.items():
                if filter_value is not None:
                    # 这里可以根据需要扩展更复杂的过滤逻辑
                    pass
        
        # 3. 获取总数（在应用排序和分页之前）
        from sqlalchemy import select
        count_query = select(func.count()).select_from(query.subquery())
        total = await session.scalar(count_query) or 0
        
        # 4. 应用排序
        if params.sort and params.sort.sort_by and sort_mapping:
            sort_field = sort_mapping.get(params.sort.sort_by)
            if sort_field is not None:
                if params.sort.sort_order == 'asc':
                    query = query.order_by(asc(sort_field))
                else:
                    query = query.order_by(desc(sort_field))
        elif default_sort is not None:
            query = query.order_by(default_sort)
        
        # 5. 应用分页
        offset = (params.pagination.page - 1) * params.pagination.page_size
        query = query.offset(offset).limit(params.pagination.page_size)
        
        # 6. 执行查询
        result = await session.execute(query)
        items = result.scalars().all()
        
        return items, total
    
    @staticmethod
    def paginate_list(
        items: List[Any],
        page: int,
        page_size: int
    ) -> tuple[List, int]:
        """
        对内存中的列表应用分页
        
        Args:
            items: 要分页的列表
            page: 页码
            page_size: 每页大小
            
        Returns:
            tuple: (paginated_items, total_count)
        """
        total = len(items)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        return items[start_idx:end_idx], total
    
    @staticmethod
    def build_response(
        items: List[T],
        total: int,
        page: int,
        page_size: int
    ) -> PaginatedResponse[T]:
        """构建分页响应"""
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        return PaginatedResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,  # 使用请求的page_size，而不是实际返回的数量
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )

    @staticmethod
    def apply_search_filters(
        query: Select,
        search_params: SearchParams,
        search_fields: Dict[str, Any]
    ) -> Select:
        """应用搜索过滤器"""
        if not search_params.search or not search_fields:
            return query
            
        search_conditions = []
        search_term = f"%{search_params.search}%"
        
        # 如果指定了搜索字段，只在这些字段中搜索
        if search_params.search_fields:
            for field_name in search_params.search_fields:
                if field_name in search_fields:
                    search_conditions.append(search_fields[field_name].ilike(search_term))
        else:
            # 否则在所有可搜索字段中搜索
            for field in search_fields.values():
                search_conditions.append(field.ilike(search_term))
        
        if search_conditions:
            query = query.where(or_(*search_conditions))
            
        return query

    @staticmethod
    def apply_custom_filters(
        query: Select,
        filters: Dict[str, Any]
    ) -> Select:
        """应用自定义过滤器 - 子类可以重写此方法"""
        # 基础实现，子类可以根据需要扩展
        return query
