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

import os

from langchain_huggingface import HuggingFaceEmbeddings

if __name__ == "__main__":
    repo_is = [
        "BAAI/bge-large-zh",
    ]
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ.setdefault("https_proxy", "socks5://127.0.0.1:34001")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
    )

    # from text2vec import SentenceModel
    # model_name = "GanymedeNil/text2vec-large-chinese"
    # model = SentenceModel(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # for repo_id in repo_is:
    #     print("repo:", repo_id)
    #     snapshot_download(repo_id=repo_id)
