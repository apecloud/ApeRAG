import asyncio
import os

from elasticsearch import Elasticsearch

from aperag.context.full_text import insert_document, search_document

es = Elasticsearch(hosts="http://127.0.0.1:9200")

index_name = "collection-12345"


def prepare_datasets(test_cases_path):
    test_cases = []
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(test_cases_path):
        for file in f:
            files.append(os.path.join(r, file))
    for idx, file in enumerate(files):
        name = os.path.basename(file)
        with open(file) as f:
            data = f.read()
        insert_document(index_name, idx, name, data)
    return test_cases


# prepare_datasets("/Users/ziang/git/ApeRAG/resources/documents/tos-feishu-parser-markdown")


def prepare_queries(test_cases_path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(test_cases_path):
        for file in f:
            files.append(file)
    return files


path = "/Users/ziang/git/ApeRAG/resources/datasets/tos"
queries = prepare_queries(path)
for q in queries:
    resp = es.indices.analyze(index=index_name, body={"text": q}, analyzer="ik_smart")
    tokens = []
    start_offset = 0
    end_offset = 0
    for token in resp.body["tokens"]:
        tokens.append(token["token"])
    result = " ".join(tokens)
    print(f"[{q}]: {result}")

# search_document(["v1.0.2"])
# search_document(["功能", "修复"])
# search_document(["功能", "修复", "阻", "试"])
# print(search_document(col_id, ["TOS", "延迟", "启动"]))
# print(search_document(col_id, ["延迟", "启动"]))
# print(search_document(col_id, ["延迟", "启"]))
# print(search_document(col_id, ["延迟启动"]))
# print(search_document(col_id, ["介绍", "延迟启动"]))
task = asyncio.create_task(search_document(index_name, []))
print(task.result())
# tokens = es.indices.analyze(index="collection-2", body={"text": content}, analyzer="ik_max_word")
