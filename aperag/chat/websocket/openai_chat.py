#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import requests


def openai_generate_stream(params):
    history = []
    prompt = params["prompt"]
    stop = params.get("stop", "###")

    headers = {
        "Authorization": "Bearer " + params.get('proxy_api_key'),
        "Token": params.get('proxy_api_key'),
    }

    messages = prompt.split(stop)
    # Add history conversation
    for message in messages:
        if len(message) <= 0:
            continue
        history.append(
            {
                "role": "system",
                "content": message,
            }
        )

    payloads = {
        "model": "gpt-3.5-turbo",
        "messages": history,
        "temperature": params.get("temperature"),
        "max_tokens": params.get("max_new_tokens"),
        "stream": True,
    }

    res = requests.post(
        params.get('proxy_server_url'), headers=headers, json=payloads, stream=True
    )

    text = ""
    for line in res.iter_lines():
        if line:
            json_data = line.split(b": ", 1)[1]
            decoded_line = json_data.decode("utf-8")
            if decoded_line.lower() != "[DONE]".lower():
                obj = json.loads(json_data)
                if obj["choices"][0]["delta"].get("content") is not None:
                    content = obj["choices"][0]["delta"]["content"]
                    text += content
                    yield content
