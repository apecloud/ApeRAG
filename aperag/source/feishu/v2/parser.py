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

import io
from abc import ABC, abstractmethod

import pytablewriter
from pydantic import BaseModel

feishu_lang_code_mapping = {
    1: "plaintext",
    2: "abap",
    3: "ada",
    4: "apache",
    5: "apex",
    6: "assembly",
    7: "bash",
    8: "csharp",
    9: "c++",
    10: "c",
    11: "cobol",
    12: "css",
    13: "coffeescript",
    14: "d",
    15: "dart",
    16: "delphi",
    17: "django",
    18: "dockerfile",
    19: "erlang",
    20: "fortran",
    21: "foxpro",
    22: "go",
    23: "groovy",
    24: "html",
    25: "htmlbars",
    26: "http",
    27: "haskell",
    28: "json",
    29: "java",
    30: "javascript",
    31: "julia",
    32: "kotlin",
    33: "latex",
    34: "lisp",
    35: "logo",
    36: "lua",
    37: "matlab",
    38: "makefile",
    39: "markdown",
    40: "nginx",
    41: "objective",
    42: "openedgeabl",
    43: "php",
    44: "perl",
    45: "postscript",
    46: "power",
    47: "prolog",
    48: "protobuf",
    49: "python",
    50: "r",
    51: "rpg",
    52: "ruby",
    53: "rust",
    54: "sas",
    55: "scss",
    56: "sql",
    57: "scala",
    58: "scheme",
    59: "scratch",
    60: "shell",
    61: "swift",
    62: "thrift",
    63: "typescript",
    64: "vbscript",
    65: "visual",
    66: "xml",
    67: "yaml",
    68: "cmake",
    69: "diff",
    70: "gherkin",
    71: "graphql",
    72: "opengl shading language",
    73: "properties",
    74: "solidity",
    75: "toml"
}


class UserAccessToken(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    name: str
    en_name: str
    avatar_url: str
    avatar_thumb: str
    avatar_middle: str
    avatar_big: str
    open_id: str
    union_id: str
    email: str
    enterprise_email: str
    user_id: str
    mobile: str
    tenant_key: str
    refresh_expires_in: int
    refresh_token: str
    sid: str


class FeishuBlockParser(ABC):
    def __init__(self, doc_id, blocks):
        self.block_map = {}
        self.doc_id = doc_id
        for block in blocks:
            self.block_map[block["block_id"]] = block

    def gen(self):
        return self.handle_block(self.block_map[self.doc_id])

    @abstractmethod
    def handle_block(self, block, indent=0):
        pass

    def parse_table(self, table):
        rows = []
        for index, block_id in enumerate(table["cells"]):
            cell_block = self.block_map[block_id]
            cell_content = self.handle_block(cell_block)
            cell_content = cell_content.replace("\n", "<br/>")
            row_index = int(index / table["property"]["column_size"])
            if len(rows) < row_index + 1:
                rows.append([])
            rows[row_index].append(cell_content)
        text = self.renderMarkdownTable(rows)
        text += "\n"
        return text

    def parse_table_cell(self, block):
        text = ""
        for child in block["children"]:
            child_block = self.block_map[child]
            text += self.handle_block(child_block)
        return text

    def parse_block_page(self, block):
        text = ""
        text += self.parse_block_text(block["page"]["elements"], [])

        for child in block.get("children", []):
            child_block = self.block_map[child]
            text += self.handle_block(child_block)
        return text

    def parse_block_text(self, elements, children):
        text = ""
        inline = len(elements) > 1
        for element in elements:
            text += self.parse_block_text_element(element, inline)
        text += "\n"
        for child in children:
            child_block = self.block_map[child]
            text += self.handle_block(child_block)
        return text

    def parse_block_bullet_list(self, block, indent):
        text = "- "
        text += self.parse_block_text(block["bullet"]["elements"], [])

        for child_id in block.get("children", []):
            child_block = self.block_map[child_id]
            text += self.handle_block(child_block, indent + 1)
        return text

    def parse_block_ordered_list(self, block, indent):
        order = 1
        parent = self.block_map[block["parent_id"]]
        for index, child in enumerate(parent["children"]):
            if child != block["block_id"]:
                continue
            i = index - 1
            while i >= 0:
                if self.block_map[parent["children"][i]]["block_type"] == 13:
                    order += 1
                else:
                    break
                i -= 1
            break

        text = f"{order}. "
        text += self.parse_block_text(block["ordered"]["elements"], [])

        for child_id in block.get("children", []):
            child_block = self.block_map[child_id]
            text += self.handle_block(child_block, indent + 1)
        return text

    @staticmethod
    def parse_block_text_element(element, inline):
        text = ""
        text_run = element.get("text_run", "")
        if text_run:
            text += text_run["content"]

        mention_user = element.get("mention_user", "")
        if mention_user:
            text += mention_user["user_id"]

        mention_doc = element.get("mention_doc", "")
        if mention_doc:
            text += "[%s](%s)" % (mention_doc.get("title", ""), mention_doc.get("url", ""))

        equation = element.get("equation", "")
        if equation:
            symbol = "$$"
            if inline:
                symbol = "$"
            text += "%s%s%s" % (symbol, equation, symbol)
        return text

    @staticmethod
    def renderMarkdownTable(data):
        writer = pytablewriter.MarkdownTableWriter()
        writer.stream = io.StringIO()
        writer.headers = data[0]
        writer.value_matrix = data[1:]
        writer.write_table()
        return writer.stream.getvalue()


class Feishu2Markdown(FeishuBlockParser):

    # https://open.feishu.cn/document/ukTMukTMukTM/uUDN04SN0QjL1QDN/document-docx/docx-v1/data-structure/block#e8ce4e8e
    def handle_block(self, block, indent=0):
        text = "\t" * indent
        block_type = block["block_type"]
        match block_type:
            case 1:
                # page
                text += "# "
                text += self.parse_block_page(block)
            case 2:
                # text
                text += self.parse_block_text(block["text"]["elements"], block.get("children", []))
            case 3:
                # heading 1
                text += "# "
                text += self.parse_block_text(block["heading1"]["elements"], block.get("children", []))
            case 4:
                # heading 2
                text += "## "
                text += self.parse_block_text(block["heading2"]["elements"], block.get("children", []))
            case 5:
                # heading 3
                text += "### "
                text += self.parse_block_text(block["heading3"]["elements"], block.get("children", []))
            case 6:
                # heading 4
                text += "#### "
                text += self.parse_block_text(block["heading4"]["elements"], block.get("children", []))
            case 7:
                # heading 5
                text += "##### "
                text += self.parse_block_text(block["heading5"]["elements"], block.get("children", []))
            case 8:
                # heading 6
                text += "###### "
                text += self.parse_block_text(block["heading6"]["elements"], block.get("children", []))
            case 9:
                # heading 7
                text += "####### "
                text += self.parse_block_text(block["heading7"]["elements"], block.get("children", []))
            case 10:
                # heading 8
                text += "######## "
                text += self.parse_block_text(block["heading8"]["elements"], block.get("children", []))
            case 11:
                # heading 9
                text += "######### "
                text += self.parse_block_text(block["heading9"]["elements"], block.get("children", []))
            case 12:
                # bullet
                text += self.parse_block_bullet_list(block, indent)
            case 13:
                # ordered list
                text += self.parse_block_ordered_list(block, indent)
                text += "\n"
            case 14:
                # code
                lang = feishu_lang_code_mapping[block["code"]["style"]["language"]]
                text += f"```{lang}\n"
                text += self.parse_block_text(block["code"]["elements"], block.get("children", []))
                text += "```\n"
            case 15:
                # quote
                text += "> "
                text += self.parse_block_text(block["quote"]["elements"], block.get("children", []))
            case 17:
                # todo
                if block["todo"]["style"]["done"]:
                    text += "- [x] "
                else:
                    text += "- [ ] "
                text += self.parse_block_text(block["todo"]["elements"], block.get("children", []))
            case 19:
                # callout
                text += self.parse_block_text([], block.get("children", []))
            case 22:
                # divider
                text += "---\n"
            case 24:
                # grid
                for child in block.get("children", []):
                    child_block = self.block_map[child]
                    text += self.handle_block(child_block)
            case 25:
                # grid column
                for child in block.get("children", []):
                    child_block = self.block_map[child]
                    text += self.handle_block(child_block)
            case 31:
                # table
                text += self.parse_table(block["table"])
            case 32:
                # table cell
                text += self.parse_table_cell(block)
            case 34:
                # quote container
                for child in block.get("children", []):
                    child_block = self.block_map[child]
                    text += "> "
                    text += self.handle_block(child_block)
            case _:
                print(f"Unhandled block type {block_type}")
                print(block)
        return text


class Feishu2PlainText(FeishuBlockParser):

    # https://open.feishu.cn/document/ukTMukTMukTM/uUDN04SN0QjL1QDN/document-docx/docx-v1/data-structure/block#e8ce4e8e
    def handle_block(self, block, indent=0):
        text = "\t" * indent
        block_type = block["block_type"]
        match block_type:
            case 1:
                # page
                text += self.parse_block_page(block)
            case 2:
                # text
                text += self.parse_block_text(block["text"]["elements"], block.get("children", []))
            case 3:
                # heading 1
                text += self.parse_block_text(block["heading1"]["elements"], block.get("children", []))
            case 4:
                # heading 2
                text += self.parse_block_text(block["heading2"]["elements"], block.get("children", []))
            case 5:
                # heading 3
                text += self.parse_block_text(block["heading3"]["elements"], block.get("children", []))
            case 6:
                # heading 4
                text += self.parse_block_text(block["heading4"]["elements"], block.get("children", []))
            case 7:
                # heading 5
                text += self.parse_block_text(block["heading5"]["elements"], block.get("children", []))
            case 8:
                # heading 6
                text += self.parse_block_text(block["heading6"]["elements"], block.get("children", []))
            case 9:
                # heading 7
                text += self.parse_block_text(block["heading7"]["elements"], block.get("children", []))
            case 10:
                # heading 8
                text += self.parse_block_text(block["heading8"]["elements"], block.get("children", []))
            case 11:
                # heading 9
                text += self.parse_block_text(block["heading9"]["elements"], block.get("children", []))
            case 12:
                # bullet
                text += self.parse_block_bullet_list(block, indent)
            case 13:
                # ordered list
                text += self.parse_block_ordered_list(block, indent)
                text += "\n"
            case 14:
                # code
                text += self.parse_block_text(block["code"]["elements"], block.get("children", []))
            case 15:
                # quote
                text += self.parse_block_text(block["quote"]["elements"], block.get("children", []))
            case 17:
                # todo
                text += self.parse_block_text(block["todo"]["elements"], block.get("children", []))
            case 19:
                # callout
                text += self.parse_block_text([], block.get("children", []))
            case 22:
                # divider
                text += "---\n"
            case 24:
                # grid
                for child in block.get("children", []):
                    child_block = self.block_map[child]
                    text += self.handle_block(child_block)
            case 25:
                # grid column
                for child in block.get("children", []):
                    child_block = self.block_map[child]
                    text += self.handle_block(child_block)
            case 31:
                # table
                text += self.parse_table(block["table"])
            case 32:
                # table cell
                text += self.parse_table_cell(block)
            case 34:
                # quote container
                for child in block.get("children", []):
                    child_block = self.block_map[child]
                    text += self.handle_block(child_block)
            case _:
                print(f"Unhandled block type {block_type}")
                print(block)
        return text

