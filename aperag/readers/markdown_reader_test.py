from pathlib import Path

from .markdown_reader import MarkdownReader

f = "/Users/ziang/git/ApeRAG/resources/documents/tos-feishu-parser-markdown/4.8  延迟启动v80s3qlh.md"


docs = MarkdownReader().load_data(Path(f))
for doc in docs:
    print(doc.text)

