from pathlib import Path

from docx_reader import MyDocxReader

test_my_docx_reader = MyDocxReader()

res = test_my_docx_reader.load_data(Path("/Users/alal/DeepRAG/tests/reader_test/docx_reader/test-2.docx"))

print(res)
