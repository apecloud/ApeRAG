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

from django.contrib import admin
from django.utils.html import format_html

from aperag.store.chat import Chat
from aperag.store.collection import Collection
from aperag.store.document import Document
from aperag.store.collection_sync_history import CollectionSyncHistory

admin.site.site_header = 'ApeRAG admin'  # set header
admin.site.site_title = 'ApeRAG admin'  # set title
admin.site.index_title = 'ApeRAG admin'


class DocumentInline(admin.StackedInline):
    model = Document


class ChatInline(admin.StackedInline):
    model = Chat


@admin.register(Collection)
class CollectionAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'description', 'user', 'status', 'type', 'created_date', 'updated_date')
    list_filter = ('status',)
    search_fields = ('title',)
    inlines = [DocumentInline]  # Associated models displayed inline
    '''Replacement value for empty field'''
    empty_value_display = 'NA'

    def created_date(self, obj):
        return format_html(
            '<span style="color: black;">{}</span>',
            obj.gmt_updated.strftime("%Y-%m-%d %H:%M:%S")
        )

    created_date.short_description = 'created'

    def updated_date(self, obj):
        return format_html(
            '<span style="color: black;">{}</span>',
            obj.gmt_updated.strftime("%Y-%m-%d %H:%M:%S")
        )

    updated_date.short_description = 'updated'


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'status', 'collection_id', 'created_date', 'updated_date')
    list_filter = ('status',)
    search_fields = ('name', 'user')
    empty_value_display = 'NA'

    def created_date(self, obj):
        return format_html(
            '<span style="color: black;">{}</span>',
            obj.gmt_updated.strftime("%Y-%m-%d %H:%M:%S")
        )

    created_date.short_description = 'created'

    def updated_date(self, obj):
        return format_html(
            '<span style="color: black;">{}</span>',
            obj.gmt_updated.strftime("%Y-%m-%d %H:%M:%S")
        )

    updated_date.short_description = 'updated'


# admin.site.register(Chat)
@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ('user', 'status', 'collection', 'gmt_created', 'gmt_updated')
    list_filter = ['status']
    search_fields = ('user',)
    '''Replacement value for empty field'''
    empty_value_display = 'NA'

    def collection(self, obj):
        return obj.collection.title

    collection.admin_order_field = 'collection__title'


@admin.register(CollectionSyncHistory)
class SyncHistoryAdmin(admin.ModelAdmin):
    list_display = (
    'id', 'user', 'collection', 'total_documents', 'new_documents', 'deleted_documents', 'processing_documents', 'modified_documents',
    'failed_documents', 'successful_documents', 'total_documents_to_sync', 'start_time', 'execution_time')
