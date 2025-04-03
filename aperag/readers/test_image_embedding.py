import json
import os
import time

import requests
from langchain import PromptTemplate

from deeprag.query.query import QueryWithEmbedding, get_packed_answer
from deeprag.readers.base_embedding import get_embedding_model
from deeprag.readers.local_path_embedding import LocalPathEmbedding
from deeprag.utils.date import elapsed_time
from deeprag.vectorstore.connector import VectorStoreConnectorAdaptor

# CFG = Config()

VICUNA_REFINE_TEMPLATE = (
    "### Human:\n"
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "Given the new context, refine and synthesize the original answer to better \n"
    "answer the question. Make sure that the refine answer is less than 200 words. \n"
    "### Assistant:"
)

VICUNA_QA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def test_local_path_embedding(path: str, collection_name: str):
    ctx = {
        "url": "http://localhost",
        "port": 6333,
        "collection": collection_name,
        "vector_size": 768,
        "distance": "Cosine",
        "timeout": 1000,
    }
    adaptor = VectorStoreConnectorAdaptor("qdrant", ctx)
    lpm = LocalPathEmbedding(adaptor, {"model_type": "huggingface"}, input_dir=path)
    lpm.load_data()


def test_local_path_embedding_query(query: str, collection_name: str):
    ctx = {
        "url": "http://localhost",
        "port": 6333,
        "collection": collection_name,
        "vector_size": 768,
        "distance": "Cosine",
        "timeout": 1000,
    }
    adaptor = VectorStoreConnectorAdaptor("qdrant", ctx)
    embedding, _ = get_embedding_model("huggingface")
    vector = embedding.get_query_embedding(query)
    query_embedding = QueryWithEmbedding(query=query, top_k=3, embedding=vector)

    results = adaptor.connector.search(
        query_embedding,
        collection_name=collection_name,
        query_vector=query_embedding.embedding,
        with_vectors=True,
        limit=query_embedding.top_k,
    )

    print("hits:", results)


def get_streaming_response(llm_server: str, input: str):
    response = requests.post("%s/generate_stream" % llm_server, json=input, stream=True)
    for line in response.iter_content():
        if line:
            content = line.decode("utf-8")
            yield content


def stream(llm_server: str, input: str):
    try:
        s = requests.Session()
        r = s.post("%s/generate_stream" % llm_server, json=input, stream=True)
        buffer = ""

        # Loop and update text
        for c in r.iter_content(chunk_size=1):
            if c == b"\x00":
                continue

            c = c.decode("utf-8")
            buffer += c

            if "}" in c:
                j = json.loads(buffer)
                buffer = ""
                print(j["text"])
    finally:
        pass


def test_local_llm_qa(query: str, collection_name: str):
    ctx = {
        "url": "http://localhost",
        "port": 6333,
        "collection": collection_name,
        "vector_size": 768,
        "distance": "Cosine",
        "timeout": 1000,
    }
    adaptor = VectorStoreConnectorAdaptor("qdrant", ctx)

    embedding, _ = get_embedding_model("huggingface")
    vector = embedding.get_query_embedding(query)
    query_embedding = QueryWithEmbedding(query=query, top_k=10, embedding=vector)

    hits = adaptor.connector.search(
        query_embedding,
        collection_name=collection_name,
        query_vector=query_embedding.embedding,
        with_vectors=True,
        limit=query_embedding.top_k,
    )

    answer_text = get_packed_answer(hits.results, 1900)

    prompt = PromptTemplate.from_template(VICUNA_REFINE_TEMPLATE)
    prompt_str = prompt.format(query_str=query, existing_answer=answer_text)

    # prompt = PromptTemplate.from_template(DEFAULT_TEXT_QA_PROMPT_TMPL)
    # prompt_str = prompt.format(query_str=query, context_str=context_msg)
    input = {
        "prompt": prompt_str,
        "temperature": 0,
        "max_new_tokens": 2048,
        "model": "gptj-6b",
        "stop": "### Assistant:",
    }

    print("prompt ", prompt_str)
    print(len(prompt_str))

    start = time.time()
    llm_server = "http://47.96.65.19:8000"

    # for line in get_streaming_response(llm_server, input):
    # run stream in a real terminal is ok, for emulator terminal, it is not well adapated
    stream(llm_server=llm_server, input=input)

    elapsed = elapsed_time(start)
    print("elapsed time ", elapsed)


def start():
    # test_local_path_embedding()
    # test_local_path_embedding_query("what is data lake")
    # test_local_llm_qa("what is data lake", "paper")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    test_local_path_embedding(
        path="../resources/datasets/menu_pics", collection_name="pics2"
    )
    test_local_llm_qa("any menu about shrimp?", "pics2")


if __name__ == "__main__":
    start()
