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


import requests
from bs4 import BeautifulSoup


def web_crawler(url):
    # Send a GET request to the specified URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all text elements within the HTML
        text_elements = soup.find_all(string=True)

        # Filter out unnecessary elements (e.g., scripts, styles, comments)
        filtered_text = [
            element.strip()
            for element in text_elements
            if element.parent.name not in ["script", "style", "head", "title", "meta", "[document]"]
        ]

        # Join the filtered text elements into a single string
        text_content = " ".join(filtered_text)

        # Print the extracted text content
        return text_content
    else:
        print("Failed to retrieve the webpage.")


def get_zhihu(url):  # Replace with the desired webpage URL
    question_idx = url.find("question")
    context = web_crawler(url)
    if question_idx != -1:
        question_start = context.find("登录/注册")
        question_end = context.find(" ​ 关注者")
        if question_end == -1:
            question_end = question_end = context.find(" 关注者 ")
        question = context[question_start + 5 : question_end]
        # print(question+"\n")

        answer_start = context.find(" 默认排序 ")
        answer = context[answer_start + 5 :]
        print(answer)
        return question + answer
    return context
