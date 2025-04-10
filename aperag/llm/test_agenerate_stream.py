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

import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch

import aiohttp
from aioresponses import aioresponses
from base import KubeBlocksLLMPredictor

from aperag.llm.baichuan import BaiChuanPredictor
from aperag.llm.custom import CustomLLMPredictor
from aperag.llm.openai import OpenAIPredictor
from aperag.llm.wenxin import BaiduQianFan


class TestBaiChuanPredictor(unittest.IsolatedAsyncioTestCase):

    async def test_stream_async_behabior(self):
        predictor1 = BaiChuanPredictor(api_key="mock_api_key",
                                       secret_key="mock_secret_key",
                                       url="http://192.0.2.0")  # 设置为一个不可访问的URL

        predictor2 = BaiChuanPredictor(api_key="mock_api_key",
                                       secret_key="mock_secret_key",
                                       url="https://api.baichuan-ai.com/v1/stream/chat")

        task_order_log = []

        async def async_task():

            try:
                _ = [tokens async for tokens in predictor1.agenerate_stream("test")]
            except aiohttp.ClientConnectorError:
                pass
            task_order_log.append("task1 completed.")  # 在尝试连接多次(超过60s)后结束，打印log信息

        async def test_stream_behavior():

            mock_response = {
                "code": 0,
                "data": {
                    "messages": [
                        {"content": "Kubernetes\u7684\u6838\u5fc3\u6280\u672f"},
                        {"content": "Service\u7684\u4f5c\u7528\u662f\u9632\u6b62Pod"},
                        {"content": "\u5931\u8054\uff08\u670d\u52a1\u53d1\u73b0\uff09"},
                        {"content": "\u548c\u5b9a\u4e49Pod\u8bbf\u95ee\u7b56\u7565"},
                        {"content": "\uff08\u8d1f\u8f7d\u5747\u8861\uff09\u3002"},
                    ]
                }
            }

            mock_responses = [
                "Kubernetes的核心技术",
                "Service的作用是防止Pod",
                "失联（服务发现）",
                "和定义Pod访问策略",
                "（负载均衡）。"
            ]

            prompt = "Kubernetes的pod是什么"

            with aioresponses() as m:
                m.post(predictor2.url, status=200, payload=mock_response)

                messages = []
                async for message in predictor2.agenerate_stream(prompt=prompt):
                    messages.append(message)

            self.assertEqual(messages, mock_responses)

            task_order_log.append("task2 completed.")  # 任务执行结束，打印日志信息

        # 使用gather同时启动两个任务
        # 如果agenerate_stream是异步的，那么在task1多次尝试连接期间，task2就已经在执行中了
        # 那么task2一定比task1先结束： (1和2同时开始)---2结束--------1结束

        _, _ = await asyncio.gather(async_task(), test_stream_behavior())

        self.assertEqual(task_order_log, ["task2 completed.", "task1 completed."])


class TestOpenAIPredictor(unittest.IsolatedAsyncioTestCase):

    async def test_stream_async_behabior(self):
        predictor1 = OpenAIPredictor(
            token="YOUR_TEST_TOKEN",
            model="gpt-3.5-turbo",
            endpoint="http://192.0.2.0"
        )
        predictor2 = OpenAIPredictor(
            token="YOUR_TEST_TOKEN",
            model="gpt-3.5-turbo",
            endpoint="https://api.openai.com/v1"
        )

        task_order_log = []

        async def async_task():
            try:
                _ = [tokens async for tokens in predictor1.agenerate_stream("test")]
            except Exception:
                pass
            task_order_log.append("task1 completed.")  # 在尝试连接多次(超过60s)后结束，打印log信息

        async def test_stream_behavior():
            mock_response = [
                {"choices": [{"finish_reason": "continue", "delta": {"content": "Kubernetes是"}}]},
                {"choices": [{"finish_reason": "continue", "delta": {"content": "一个开源的"}}]},
                {"choices": [{"finish_reason": "continue", "delta": {"content": "容器编排平台，"}}]},
                {"choices": [{"finish_reason": "continue", "delta": {"content": "用于自动化应用程序的"}}]},
                {"choices": [{"finish_reason": "continue", "delta": {"content": "部署、扩展和管理。"}}]},
                {"choices": [{"finish_reason": "stop", "delta": {"content": ""}}]}
            ]

            expected_responses = [
                "Kubernetes是",
                "一个开源的",
                "容器编排平台，",
                "用于自动化应用程序的",
                "部署、扩展和管理。"
            ]

            prompt = "Kubernetes是什么"

            async def async_generator():
                for msg in mock_response:
                    yield msg

            with patch('predict.openai.ChatCompletion.acreate', return_value=async_generator()):
                results = [content async for content in predictor2.agenerate_stream(prompt=prompt)]
                self.assertEqual(results, expected_responses)

            task_order_log.append("task2 completed.")  # 任务开始执行2s后结束，打印日志信息

        # 使用gather同时启动两个任务
        # 如果agenerate_stream是异步的，那么在task1多次尝试连接期间，task2就已经在执行中了
        # 那么task2一定比task1先结束： (1和2同时开始)---2结束--------1结束

        _, _ = await asyncio.gather(async_task(), test_stream_behavior())

        self.assertEqual(task_order_log, ["task2 completed.", "task1 completed."])


class TestKubeBlocksLLMPredictor(unittest.IsolatedAsyncioTestCase):

    async def test_stream_async_behavior(self):
        predictor1 = KubeBlocksLLMPredictor(endpoint="http://192.0.2.0")
        predictor2 = KubeBlocksLLMPredictor(endpoint="http://localhost:18000")

        task_order_log = []

        async def async_task():
            try:
                _ = [tokens async for tokens in predictor1.agenerate_stream("test")]
            except Exception:
                pass

            await asyncio.sleep(60)  # KubeBlocksLLMPredictor尝试连接的时长太短，用睡眠模拟更长一些

            task_order_log.append("task1 completed.")  # 在尝试连接多次(超过60s)后结束，打印log信息

        async def test_stream_behavior():
            # 模拟响应数据
            mock_response = MagicMock()
            mock_response.iter_lines.return_value = [
                b'{"text": ["Hello, nice to meet you!"]}',
                b'{"text": ["Hello, nice to meet you! How can I help?"]}'
            ]

            mock_responses = [", nice to meet you!",
                              " How can I help?"
                              ]

            prompt = "Hello"

            # 使用patch模拟requests.post调用
            with patch('predict.requests.post', return_value=mock_response):

                async_result = [tokens async for tokens in predictor2.agenerate_stream(prompt)]
                self.assertEqual(len(async_result), len(mock_responses))
                self.assertEqual(async_result, mock_responses)

            task_order_log.append("task2 completed.")

        # 使用gather同时启动两个任务
        # 如果agenerate_stream是异步的，那么在task1多次尝试连接期间，task2就已经在执行中了
        # 那么task2一定比task1先结束： (1和2同时开始)---2结束--------1结束

        _, _ = await asyncio.gather(async_task(), test_stream_behavior())

        self.assertEqual(task_order_log, ["task2 completed.", "task1 completed."])


class TestCustomLLMPredictor(unittest.IsolatedAsyncioTestCase):

    async def test_stream_async_behabior(self):
        predictor1 = CustomLLMPredictor(model="test_model", endpoint="http://192.0.2.0")
        predictor2 = CustomLLMPredictor(model="test_model", endpoint="http://localhost:18000")

        task_order_log = []

        async def async_task():
            try:
                _ = [tokens async for tokens in predictor1.agenerate_stream("test")]
            except aiohttp.ClientConnectorError:
                pass
            task_order_log.append("task1 completed.")  # 在尝试连接多次(超过60s)后结束，打印log信息

        async def test_stream_behavior():
            mock_data = [
                {"text": "Hello, this is the first line."},
                {"text": "And here is the second."}
            ]

            mock_responses = ["Hello, this is the first line.",
                              "And here is the second."
                              ]

            mock_response = b"\0".join([json.dumps(item).encode('utf-8') for item in mock_data])

            prompt = "test"

            with aioresponses() as mocked:
                mocked.post(predictor2.url, status=200, body=mock_response)

                results = [res async for res in predictor2.agenerate_stream(prompt=prompt)]

            self.assertEqual(results, mock_responses)

            task_order_log.append("task2 completed.")  # 任务执行结束，打印日志信息

        # 使用gather同时启动两个任务
        # 如果agenerate_stream是异步的，那么在task1多次尝试连接期间，task2就已经在执行中了
        # 那么task2一定比task1先结束： (1和2同时开始)---2结束--------1结束

        _, _ = await asyncio.gather(async_task(), test_stream_behavior())

        self.assertEqual(task_order_log, ["task2 completed.", "task1 completed."])


class TestBaiduQianFan(unittest.IsolatedAsyncioTestCase):

    async def test_agenerate_stream_async_effect(self):
        predictor1 = BaiduQianFan(api_key="mock_api_key", secret_key="mock_secret_key", endpoint="http://192.0.2.0")
        predictor2 = BaiduQianFan(api_key="mock_api_key", secret_key="mock_secret_key")

        task_order_log = []

        async def async_task():
            try:
                _ = [tokens async for tokens in predictor1.agenerate_stream("test")]
            except Exception:
                await asyncio.sleep(60)

            task_order_log.append("task1 completed.")  # 在尝试连接多次(超过60s)后结束，打印log信息

        async def test_stream_behavior():
            # 模拟响应数据
            mock_ado_response = MagicMock()
            mock_ado_response.__aiter__.return_value = [
                {"result": "以下是一些适合自驾游的路线推荐：\n\n1."},
                {
                    "result": "中国大陆最美的景观大道：川藏线，从成都出发，沿着川藏公路一路向西，经过稻城亚丁、理塘、巴塘、芒康等美景，最终到达拉萨。"},
                {
                    "result": "\n2. 丝绸之路：这是一条贯穿中国东西部的公路，从上海出发，经过西安、兰州、乌鲁木齐等城市，最终到达喀什。"},
                {"result": "沿途可以欣赏到中国北方和南方的不同景色。"},
            ]

            mock_responses = ["以下是一些适合自驾游的路线推荐：\n\n1.",
                              "中国大陆最美的景观大道：川藏线，从成都出发，沿着川藏公路一路向西，经过稻城亚丁、理塘、巴塘、芒康等美景，最终到达拉萨。",
                              "\n2. 丝绸之路：这是一条贯穿中国东西部的公路，从上海出发，经过西安、兰州、乌鲁木齐等城市，最终到达喀什。",
                              "沿途可以欣赏到中国北方和南方的不同景色。"
                              ]

            prompt = "给我推荐一些自驾游路线"

            # 使用patch模拟self.chat_comp.ado调用
            with patch('predict.qianfan.ChatCompletion.ado', return_value=mock_ado_response):
                async_result = [chunk async for chunk in predictor2.agenerate_stream(prompt)]
                self.assertEqual(len(async_result), len(mock_responses))
                self.assertEqual(async_result, mock_responses)

            task_order_log.append("task2 completed.")  # 任务开始执行2s后结束，打印日志信息

        # 使用gather同时启动两个任务
        # 如果agenerate_stream是异步的，那么在task1多次尝试连接期间，task2就已经在执行中了
        # 那么task2一定比task1先结束： (1和2同时开始)---2结束--------1结束

        _, _ = await asyncio.gather(async_task(), test_stream_behavior())

        self.assertEqual(task_order_log, ["task2 completed.", "task1 completed."])


if __name__ == "__main__":
    unittest.main()
