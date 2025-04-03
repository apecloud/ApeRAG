import hashlib
import json
import logging
import os
import time
from datetime import datetime

from langchain import PromptTemplate
from tabulate import tabulate

from deeprag.context.context import ContextManager
from deeprag.context.full_text import create_index, delete_index, insert_document, search_document
from deeprag.llm.custom import CustomLLMPredictor
from deeprag.llm.prompts import DEFAULT_CHINESE_PROMPT_TEMPLATE_V2
from deeprag.pipeline.keyword_extractor import IKExtractor
from deeprag.query.query import get_packed_answer
from deeprag.readers.base_embedding import get_embedding_model, get_rerank_model
from deeprag.readers.local_path_embedding import LocalPathEmbedding
from deeprag.utils.utils import generate_fulltext_index_name
from deeprag.vectorstore.connector import VectorStoreConnectorAdaptor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


VECTOR_DB_TYPE = "qdrant"
VECTOR_DB_CONTEXT = {"url": "http://127.0.0.1", "port": 6333, "distance": "Cosine", "timeout": 1000}


os.environ["RERANK_MODEL_PATH"] = "/Users/ziang/.cache/huggingface/hub/bge-reranker-large"


table_format = """
<html>
<head>
<style> 
  table {{ table-layout: fixed; border: 1px solid black; border-collapse: collapse; }}
  th {{ border: 1px solid black; border-collapse: collapse; white-space:break-spaces; overflow:hidden; position:sticky; top: 0; padding: 5px;}}
  td {{ border: 1px solid black; border-collapse: collapse; white-space:break-spaces; overflow:hidden; padding: 5px; word-wrap: break-all}}
</style>
</head>
{table}
</body></html>
"""


predictor = CustomLLMPredictor(model="baichuan-13b", endpoint="http://localhost:18000")


class TestCase:
    def __init__(self, query, expect):
        self.query = query
        self.expect = expect


class EmbeddingCtx:
    def __init__(self, collection_name, model_name):
        self.model_name = model_name
        self.model, self.vector_size = get_embedding_model(model_name)
        self.collection_name = f"{collection_name}_{model_name}_{self.vector_size}"
        ctx = VECTOR_DB_CONTEXT.copy()
        ctx["collection"] = self.collection_name
        self.ctx_manager = ContextManager(self.collection_name, self.model, VECTOR_DB_TYPE, ctx)
        self.vector_db_conn = VectorStoreConnectorAdaptor(VECTOR_DB_TYPE, ctx=ctx)

        self.qa_collection_name = f"{self.collection_name}-qa"
        qa_ctx = VECTOR_DB_CONTEXT.copy()
        qa_ctx["collection"] = self.qa_collection_name
        self.qa_ctx_manager = ContextManager(self.qa_collection_name, self.model, VECTOR_DB_TYPE, qa_ctx)
        self.qa_vector_db_conn = VectorStoreConnectorAdaptor(VECTOR_DB_TYPE, ctx=qa_ctx)

        self.index = generate_fulltext_index_name(self.collection_name)
        self.ranker = get_rerank_model()

    def load_file(self, file_path):
        meta_file = file_path + "-metadata"
        with open(meta_file) as f:
            local_doc = json.loads(f.read())
            metadata = local_doc["metadata"]

        loader = LocalPathEmbedding(input_files=[file_path], input_file_metadata_list=[metadata],
                                    embedding_model=self.model,
                                    vector_store_adaptor=self.vector_db_conn)
        ctx_ids = loader.load_data()
        if ctx_ids:
            doc_id = hashlib.md5(local_doc["name"].encode('utf-8')).hexdigest()
            with open(file_path) as f:
                doc_content = f.read()
            insert_document(self.index, doc_id, local_doc["name"], doc_content)

    def load_dir(self, path):
        self.vector_db_conn.connector.create_collection(vector_size=self.vector_size)
        self.qa_vector_db_conn.connector.create_collection(vector_size=self.vector_size)
        delete_index(self.index)
        create_index(self.index)

        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if file.endswith("-metadata"):
                    continue
                files.append(os.path.join(r, file))

        for file in files:
            self.load_file(file)

    def query(self, query, **kwargs):
        # results = self.qa_ctx_manager.query(query, score_threshold=0.8, topk=3, recall_factor=1)
        # if len(results) > 0:
        #     return results

        topk = kwargs.get("topk", 3)
        score_threshold = kwargs.get("score_threshold", 0.5)
        recall_factor = kwargs.get("recall_factor", 1)

        keywords = IKExtractor({"index_name": self.index, "es_host": "http://127.0.0.1:9200"}).extract(query)
        logger.info("[%s] extract keywords: %s", query, " | ".join(keywords))

        # find the related documents using keywords
        doc_names = {}
        if keywords:
            docs = search_document(self.index, keywords, topk * 3)
            for doc in docs:
                doc_names[doc["name"]] = doc["content"]
                logger.info("[%s] found keyword in document %s", query, doc["name"])

        candidates = []
        results = self.ctx_manager.query(query, score_threshold, topk * recall_factor)
        start = time.time()
        results = self.ranker.rank(query, results)[:topk]
        print(f"{[query]} references: {len(results)}, rerank cost {time.time() - start}\n")

        for result in results:
            if result.metadata["name"] not in doc_names:
                logger.info("[%s] ignore doc %s not match keywords", query, result.metadata["name"])
                continue
            candidates.append(result)

        # if no keywords found, fallback to using all results from embedding search
        if not doc_names:
            candidates = results

        # candidates = self.ranker.rank(query, candidates)[:topk]
        return candidates


def prepare_datasets(test_cases_path):
    test_cases = []
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(test_cases_path):
        for file in f:
            files.append(os.path.join(r, file))
    for file in files:
        query = os.path.basename(file)
        with open(file) as f:
            references = f.read()
        test_cases.append(TestCase(query, references))
    return test_cases


def prepare_documents(dataset_dir, models, reload=False):
    ctx_list = []
    for model in models:
        ctx = EmbeddingCtx(os.path.basename(dataset_dir), model)
        ctx_list.append(ctx)
        if reload:
            ctx.load_dir(dataset_dir)
    return ctx_list


def compare_models(test_cases, embedding_ctxs, **kwargs):
    table = []
    prompt_template = PromptTemplate(template=DEFAULT_CHINESE_PROMPT_TEMPLATE_V2, input_variables=["query", "context"])
    enable_inference = kwargs.get("enable_inference", False)
    for i, item in enumerate(test_cases):
        row = [item.query, item.expect]
        for ctx in embedding_ctxs:
            results = ctx.query(item.query, **kwargs)
            candidates = []
            for idx, result in enumerate(results):
                prefixes = [f"score: {result.score}"]
                if idx != result.rank_before:
                    prefixes.append(f"rerank: from {result.rank_before + 1} to {idx + 1}")
                prefix = '\n'.join(prefixes)
                candidates.append(f"{prefix}\n: {result.text}")
            row.append("\n\n".join(candidates))
            if enable_inference and len(results) > 0:
                context = get_packed_answer(results, 3500)
                prompt = prompt_template.format(query=item.query, context=context)
                response = ""
                for msg in predictor.generate_stream(prompt):
                    response += msg
                row.append(response)

        table.append(row)
    return table


def main(datasets, documents, models, reload, **kwargs):
    datasets = prepare_datasets(datasets)
    ctx_list = prepare_documents(documents, models, reload)
    table = compare_models(datasets, ctx_list, **kwargs)
    headers = ["query", "references"]
    enable_inference = kwargs.get("enable_inference", False)
    for i, ctx in enumerate(ctx_list):
        headers.append(ctx.model_name)
        if enable_inference:
            headers.append(f"{ctx.model_name}-inference")

    return headers, table


if __name__ == "__main__":
    dirname = os.environ.get("DEEPRAG_BASE_DIR", "/Users/ziang/git/deeprag")

    # datasets = "resources/datasets/tos"
    # datasets = "resources/datasets/releases"
    datasets = "resources/datasets/test1"
    # documents = "resources/documents/test"
    # documents = "resources/documents/tos-feishu-bad-cases-plain"
    # documents = "resources/documents/tos-feishu-bad-cases-markdown"
    documents = "resources/documents/tos-feishu-markdown"
    # documents = "resources/documents/releases"
    # documents = "resources/documents/tos-feishu-parser-markdown-2"
    # documents = "resources/documents/tos-short"
    # documents = "resources/documents/tos-feishu-parser-plain"
    # documents = "resources/documents/tos-feishu-parser-plain-without-category"
    # documents = "resources/documents/tos-pdf"
    # models = ["multilingual"]
    # models = ["text2vec"]

    datasets = os.path.join(dirname, datasets)
    documents = os.path.join(dirname, documents)

    models = ["bge"]
    # models = ["mt5"]
    # models = ["openai"]
    # models = ["text2vec", "bge"]
    # models = ["text2vec", "bge", "multilingual", "openai"]
    # models = ["text2vec", "bge", "piccolo"]
    # models = ["piccolo"]
    score_threshold = 0.5
    topk = 3
    recall_factor = 6
    reload = True
    enable_inference = False
    headers, table = main(datasets, documents, models, reload,
                          score_threshold=score_threshold,
                          topk=topk,
                          recall_factor=recall_factor,
                          enable_inference=enable_inference)
    result = table_format.format(table=tabulate(table, headers, tablefmt="html")).encode("utf-8")
    result_path = f'/tmp/{os.path.basename(documents)}-{datetime.utcnow().timestamp()}.html'
    with open(result_path, "wb+") as fd:
        fd.write(result)
    print(f"write output to file: {result_path}")
