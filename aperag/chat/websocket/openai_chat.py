#!/usr/bin/env python3
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

# -*- coding: utf-8 -*-

import json

import requests


def openai_generate_stream(params):
    history = []
    prompt = params["prompt"]
    stop = params.get("stop", "###")

    headers = {
        "Authorization": "Bearer " + params.get("proxy_api_key"),
        "Token": params.get("proxy_api_key"),
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

    res = requests.post(params.get("proxy_server_url"), headers=headers, json=payloads, stream=True)

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
