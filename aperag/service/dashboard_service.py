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

from django.shortcuts import render

from aperag.db import models as db_models


def dashboard_service(request):
    user_count = db_models.User.objects.count()
    collection_count = db_models.Collection.objects.count()
    document_count = db_models.Document.objects.count()
    chat_count = db_models.Chat.objects.count()
    context = {
        "user_count": user_count,
        "Collection_count": collection_count,
        "Document_count": document_count,
        "Chat_count": chat_count,
    }
    return render(request, "aperag/dashboard.html", context)
