from sqlalchemy.dialects import postgresql

from aperag.db.repositories.lightrag import LightragRepositoryMixin


class _FakeScalarResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class _FakeResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return _FakeScalarResult(self._items)


class _FakeSession:
    def __init__(self, items):
        self._items = items
        self.captured_stmt = None

    def execute(self, stmt):
        self.captured_stmt = stmt
        return _FakeResult(self._items)


class _RepoUnderTest(LightragRepositoryMixin):
    def __init__(self, items):
        self.session = _FakeSession(items)

    def _execute_query(self, query_func):
        return query_func(self.session)

    def _execute_transaction(self, operation):
        raise NotImplementedError


class _Entity:
    def __init__(self, entity_id):
        self.id = entity_id


class _Relation:
    def __init__(self, relation_id):
        self.id = relation_id


def _compile_postgres(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_query_lightrag_vdb_entity_by_chunk_ids_casts_rhs_to_varchar_array():
    repo = _RepoUnderTest([_Entity("entity-1")])

    result = repo.query_lightrag_vdb_entity_by_chunk_ids("workspace-1", ["chunk-1", "chunk-2", "chunk-1", ""])

    assert result == {"entity-1": repo.session._items[0]}
    compiled = _compile_postgres(repo.session.captured_stmt)
    assert "lightrag_vdb_entity.workspace = %(workspace_1)s" in compiled
    assert "lightrag_vdb_entity.chunk_ids && CAST(ARRAY[" in compiled
    assert " AS VARCHAR[])" in compiled
    assert compiled.count("param_") == 2


def test_query_lightrag_vdb_relation_by_chunk_ids_casts_rhs_to_varchar_array():
    repo = _RepoUnderTest([_Relation("relation-1")])

    result = repo.query_lightrag_vdb_relation_by_chunk_ids("workspace-1", ["chunk-a", "chunk-b"])

    assert result == {"relation-1": repo.session._items[0]}
    compiled = _compile_postgres(repo.session.captured_stmt)
    assert "lightrag_vdb_relation.workspace = %(workspace_1)s" in compiled
    assert "lightrag_vdb_relation.chunk_ids && CAST(ARRAY[" in compiled
    assert " AS VARCHAR[])" in compiled


def test_query_lightrag_vdb_entity_by_chunk_ids_returns_empty_without_query_for_empty_input():
    repo = _RepoUnderTest([_Entity("entity-1")])

    result = repo.query_lightrag_vdb_entity_by_chunk_ids("workspace-1", [])

    assert result == {}
    assert repo.session.captured_stmt is None
