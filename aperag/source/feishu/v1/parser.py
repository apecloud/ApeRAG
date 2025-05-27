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
import logging
from urllib.parse import unquote

import pytablewriter

logger = logging.getLogger(__name__)


class FeishuDocParser:
    def __init__(self, data):
        self.data = data

    def gen(self):
        text = ""
        text += self.parse_body(self.data["body"])
        return text

    def parse_paragraph(self, paragraph):
        text = ""
        style = paragraph.get("style")
        if style:
            heading_level = style.get("headingLevel", 0)
            style_list = style.get("list", {})
            style_quote = style.get("quote", False)
            if heading_level > 0:
                text += "#" * heading_level
                text += " "
            elif style_list:
                indent_level = style_list["indentLevel"]
                if indent_level > 1:
                    text += "\t" * (style_list["indentLevel"] - 2)
                match style_list["type"]:
                    case "number":
                        text += f"{style_list['number']}."
                    case "bullet":
                        text += "-"
                    case "checkBox":
                        text += "- []"
                    case "checkedBox":
                        text += "- [x]"
                text += " "
            elif style_quote:
                text += "> "
        for element in paragraph.get("elements", []):
            text += self.parse_paragraph_element(element)
        text += "\n"
        return text

    def parse_paragraph_element(self, element):
        match element["type"]:
            case "textRun":
                return self.parse_paragraph_text_run(element["textRun"])
            case "docsLink":
                return self.parse_paragraph_docs_link(element["docsLink"])
            case "equation":
                return self.parse_paragraph_equation(element["equation"])
            case _:
                return ""

    # unescape an url
    @staticmethod
    def parse_paragraph_text_run(text_run):
        text = ""
        post_write = ""
        style = text_run.get("style", {})
        if style:
            bold = style.get("bold", False)
            italic = style.get("italic", False)
            strike_through = style.get("strikethrough", False)
            underline = style.get("underline", False)
            code_inline = style.get("codeInline", False)
            link = style.get("link", {})
            if bold:
                text += "**"
                post_write += "**"
            elif italic:
                text += "*"
                post_write += "*"
            elif strike_through:
                text += "~~"
                post_write += "~~"
            elif underline:
                text += "<u>"
                post_write += "</u>"
            elif code_inline:
                text += "`"
                post_write += "`"
            elif link:
                text += "["
                url = unquote(link["url"])
                post_write += f"]({url})"
        text += text_run.get("text", "")
        text += post_write
        return text

    @staticmethod
    def parse_paragraph_docs_link(docs_link):
        url = docs_link["url"]
        return f"[]{url}"

    @staticmethod
    def parse_paragraph_equation(equation):
        equation = equation["equation"]
        return f"$${equation}$$"

    def parse_code(self, code):
        text = ""
        text += "```"
        text += code["language"]
        text += "\n"
        text += self.parse_body(code["body"])
        text += "```"
        text += "\n"
        return text

    @staticmethod
    def renderMarkdownTable(data):
        writer = pytablewriter.MarkdownTableWriter()
        writer.stream = io.StringIO()
        writer.headers = data[0]
        writer.value_matrix = data[1:]
        writer.write_table()
        return writer.stream.getvalue()

    def parse_table(self, table):
        rows = []
        for _, row in enumerate(table["tableRows"]):
            cells = []
            for cell in row["tableCells"]:
                cells.append(self.parse_table_cell(cell))
            rows.append(cells)
        text = self.renderMarkdownTable(rows)
        text += "\n"
        return text

    def parse_table_cell(self, cell):
        text = ""
        blocks = cell["body"]["blocks"]
        if not blocks:
            return ""
        for block in cell["body"]["blocks"]:
            text += self.parse_block(block).replace("\n", "<br/>")
        return text

    def parse_callout(self, callout):
        text = ""
        for block in callout["body"]["blocks"]:
            text += "> "
            text += self.parse_block(block)
        text += "\n"
        return text

    def parse_body(self, body):
        if not body:
            return ""

        text = ""
        for block in body["blocks"]:
            text += self.parse_block(block)
            text += "\n"
        return text

    def parse_block(self, block):
        match block["type"]:
            case "paragraph":
                return self.parse_paragraph(block["paragraph"])
            case "code":
                return self.parse_code(block["code"])
            case "table":
                return self.parse_table(block["table"])
            case "callout":
                return self.parse_callout(block["callout"])
            case _:
                logger.info("ignore unsupported block type %s", block["type"])
                return ""
