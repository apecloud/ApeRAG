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
