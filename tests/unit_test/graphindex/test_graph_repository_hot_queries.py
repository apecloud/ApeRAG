from sqlalchemy.dialects import postgresql

from aperag.db.repositories.graph import GraphRepositoryMixin


class _EmptyScalarResult:
    def __iter__(self):
        return iter([])


class _EmptyResult:
    def unique(self):
        return self

    def scalars(self):
        return _EmptyScalarResult()


class _CaptureSession:
    def __init__(self):
        self.captured_stmt = None

    def execute(self, stmt, _params=None):
        self.captured_stmt = stmt
        return _EmptyResult()

    def flush(self):
        return None


class _RepoUnderTest(GraphRepositoryMixin):
    def __init__(self):
        self.session = _CaptureSession()

    def _execute_query(self, query_func):
        return query_func(self.session)

    def _execute_transaction(self, operation):
        return operation(self.session)


def _compile_postgres(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"render_postcompile": True}))


def test_get_graph_nodes_by_source_ids_uses_array_overlap_not_like():
    repo = _RepoUnderTest()

    result = repo.get_graph_nodes_by_source_ids("workspace-1", ["chunk-1", "chunk-2", "chunk-1", ""])

    assert result == {}
    compiled = _compile_postgres(repo.session.captured_stmt)
    assert "string_to_array(lightrag_graph_nodes.source_id" in compiled
    assert "lightrag_graph_nodes.workspace = %(workspace_1)s" in compiled
    assert "&& CAST(ARRAY[" in compiled
    assert " AS VARCHAR[])" in compiled
    assert " LIKE " not in compiled


def test_get_graph_edges_by_source_ids_uses_array_overlap_not_like():
    repo = _RepoUnderTest()

    result = repo.get_graph_edges_by_source_ids("workspace-1", ["chunk-a", "chunk-b"])

    assert result == {}
    compiled = _compile_postgres(repo.session.captured_stmt)
    assert "string_to_array(lightrag_graph_edges.source_id" in compiled
    assert "lightrag_graph_edges.workspace = %(workspace_1)s" in compiled
    assert "&& CAST(ARRAY[" in compiled
    assert " AS VARCHAR[])" in compiled
    assert " LIKE " not in compiled


def test_get_graph_edges_batch_uses_rowwise_pair_lookup_not_big_or():
    repo = _RepoUnderTest()

    result = repo.get_graph_edges_batch("workspace-1", [("node-a", "node-b"), ("node-c", "node-d")])

    assert result == {
        ("node-a", "node-b"): {"weight": 0.0, "keywords": None, "description": None, "source_id": None},
        ("node-c", "node-d"): {"weight": 0.0, "keywords": None, "description": None, "source_id": None},
    }
    compiled = _compile_postgres(repo.session.captured_stmt)
    assert "(lightrag_graph_edges.source_entity_id, lightrag_graph_edges.target_entity_id) IN" in compiled
    assert " OR " not in compiled


def test_delete_graph_edges_batch_uses_rowwise_pair_lookup_not_big_or():
    repo = _RepoUnderTest()

    repo.delete_graph_edges_batch("workspace-1", [("node-a", "node-b"), ("node-c", "node-d")])

    compiled = _compile_postgres(repo.session.captured_stmt)
    assert "(lightrag_graph_edges.source_entity_id, lightrag_graph_edges.target_entity_id) IN" in compiled
    assert " OR " not in compiled
