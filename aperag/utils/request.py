import json


def get_user(request):
    return request.META.get("X-USER-ID", "")


def get_urls(request):
    body_str = request.body.decode('utf-8')

    data = json.loads(body_str)

    urls = [item['url'] for item in data['urls']]

    return urls


