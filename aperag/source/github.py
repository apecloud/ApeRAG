import logging
import os
import random
import uuid
from datetime import datetime
from typing import Any, Dict, Iterator

import git

from aperag.source.base import LocalDocument, RemoteDocument, Source

logger = logging.getLogger(__name__)


class GitHubSource(Source):
    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        github_config = ctx["github"]
        self.repo_url = github_config["repo"]
        self.branch = github_config.get("branch", "main")
        self.path = github_config.get("path", "/")
        self.tmp_dir = './documents/' + ''.join(random.sample(uuid.uuid4().hex, 16))

    def scan_documents(self) -> Iterator[RemoteDocument]:
        git.Repo.clone_from(self.repo_url, self.tmp_dir, branch=self.branch)
        full_path = os.path.join(self.tmp_dir, self.path)
        base_url = os.path.join(self.repo_url, "blob", self.branch)
        for root, dirs, files in os.walk(full_path):
            if '.git' in root:
                continue
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_stat = os.stat(file_path)
                    modified_time = datetime.utcfromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%dT%H:%M:%S")

                    doc = RemoteDocument(
                        name=file_path.replace(self.tmp_dir, ""),
                        size=file_stat.st_size,
                        metadata={"path": os.path.normpath(file_path),
                                  "url": file_path.replace(self.tmp_dir, base_url),
                                  "modified_time": modified_time}
                    )
                    yield doc
                except Exception as e:
                    logger.error(f"scanning local source {file_path} error {e}")
                    raise e

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        metadata["name"] = name
        return LocalDocument(name=name, path=metadata["path"], metadata=metadata)

    def close(self):
        pass

    def sync_enabled(self):
        return True

    def cleanup_document(self, filepath: str):
        pass
