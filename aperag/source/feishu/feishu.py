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

import datetime
import json
import logging
import time
from typing import Any, Dict, Iterator

import aperag.source.feishu.v1.parser as v1
import aperag.source.feishu.v2.parser as v2
from aperag.schema.view_models import CollectionConfig
from aperag.source.base import CustomSourceInitializationError, LocalDocument, RemoteDocument, Source
from aperag.source.feishu.client import FeishuClient
from aperag.source.utils import gen_temporary_file

logger = logging.getLogger(__name__)


class FeishuSource(Source):
    def __init__(self, ctx: CollectionConfig):
        super().__init__(ctx)
        self.client = FeishuClient(ctx)
        self.space_id = ctx.space_id
        self.root_node_id = ""
        self.method = "block_api"
        self.target_format = "md"
        try:
            self.client.get_node(self.root_node_id, timeout=3)
        except Exception:
            raise CustomSourceInitializationError(f"Error querying Feishu node {self.root_node_id}. Invalid parameter")

    def get_node_documents(self, space_id, node_token):
        node_mapping = {}

        # find parent titles from bottom to top
        def get_parent_titles(current_node):
            result = []
            while True:
                parent_node = node_mapping.get(current_node["parent_node_token"], None)
                if not parent_node:
                    break
                result.insert(0, parent_node["title"])
                current_node = parent_node
            return result

        root_node = self.client.get_node(node_token)
        # iterate the nodes in the BFS(Breadth First Search) way
        nodes = [root_node]
        for node in nodes:
            node_token = node["node_token"]
            node_mapping[node_token] = node
            if node["has_child"]:
                nodes.extend(self.client.get_space_nodes(space_id, node_token))
            if node["obj_type"] not in ("docx", "doc"):
                logger.info("ignore unsupported node type %s, %s", node["obj_type"], node["title"])
                continue

            metadata = {
                "titles": get_parent_titles(node),
                "obj_token": node["obj_token"],
                "obj_type": node["obj_type"],
                "modified_time": datetime.datetime.utcfromtimestamp(int(node["obj_edit_time"])),
            }
            doc = RemoteDocument(name=node["title"] + f".{self.target_format}", size=0, metadata=metadata)
            yield doc

    def scan_documents(self) -> Iterator[RemoteDocument]:
        return self.get_node_documents(self.space_id, self.root_node_id)

    def get_docx_content_with_block_api(self, node_id):
        blocks = self.client.get_docx_blocks(node_id)
        if self.target_format == "md":
            return v2.Feishu2Markdown(node_id, blocks).gen()
        elif self.target_format == "txt":
            return v2.Feishu2PlainText(node_id, blocks).gen()
        else:
            raise Exception(f"unsupported target format: {self.target_format}")

    def get_doc_content_with_block_api(self, node_id):
        data = json.loads(self.client.get_doc_blocks(node_id))
        if self.target_format == "md":
            return v1.FeishuDocParser(data).gen()
        else:
            raise Exception(f"unsupported target format: {self.target_format}")

    def get_content_with_export_api(self, doc_type, node_id, extension="pdf"):
        ticket = self.client.create_export_task(doc_id=node_id, doc_type=doc_type, extension=extension)
        while True:
            result = self.client.query_export_task(ticket, node_id)
            match result["job_status"]:
                case 0:  # success
                    file_token = result["file_token"]
                    break
                case 1 | 2:  # initializing, running
                    time.sleep(1)
                    pass
                case _:
                    raise Exception(f"export task failed: {result}")
        return self.client.download_doc(file_token)

    def get_docx_content(self, node_id):
        match self.method:
            case "plain_api":
                return self.client.get_docx_plain_content(node_id).encode("utf-8")
            case "block_api":
                return self.get_docx_content_with_block_api(node_id).encode("utf-8")
            case "export_api":
                return self.get_content_with_export_api(doc_type="docx", node_id=node_id)
            case _:
                raise Exception(f"unsupported method: {self.method}")

    def get_doc_content(self, node_id):
        match self.method:
            case "plain_api":
                return self.client.get_doc_plain_content(node_id).encode("utf-8")
            case "block_api":
                return self.get_doc_content_with_block_api(node_id).encode("utf-8")
            case "export_api":
                return self.get_content_with_export_api(doc_type="doc", node_id=node_id)
            case _:
                raise Exception(f"unsupported method: {self.method}")

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        node_id = metadata["obj_token"]
        match metadata["obj_type"]:
            case "docx":
                content = self.get_docx_content(node_id)
            case "doc":
                content = self.get_doc_content(node_id)
            case _:
                raise Exception(f"unsupported node type: {metadata['obj_type']}")

        temp_file = gen_temporary_file(name)
        temp_file.write(content)
        temp_file.close()
        metadata["name"] = name
        return LocalDocument(name=name, path=temp_file.name, metadata=metadata)

    def sync_enabled(self):
        return True
