"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

import kubechat.views
from kubechat.views import default_page
from django.conf.urls import handler404

handler404 = kubechat.views.default_page

urlpatterns = [
    path("api/", include("kubechat.urls")),
    path("admin/", admin.site.urls),
    path('', include('django_prometheus.urls')),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


