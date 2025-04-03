"""
ASGI config for config project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from django.urls import path, re_path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

django_asgi_app = get_asgi_application()

from aperag.auth.validator import WebSocketAuthMiddleware  # noqa: E402
from aperag.chat.sse.sse import ServerSentEventsConsumer  # noqa: E402
from aperag.chat.websocket.routing import websocket_urlpatterns  # noqa: E402

application = ProtocolTypeRouter(
    {
        "http": URLRouter([
            path('events', ServerSentEventsConsumer.as_asgi()),
            re_path(r'', get_asgi_application()),
        ]),
        "websocket": WebSocketAuthMiddleware(URLRouter(websocket_urlpatterns)),
    }
)
