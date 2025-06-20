from datetime import datetime

import pytest

from aperag.graph.lightrag.context_types import (
    LightRagEntityContext,
    LightRagRelationContext,
    LightRagTextUnitContext,
    json_list_to_entity_contexts,
    json_list_to_relation_contexts,
    json_list_to_text_unit_contexts,
    json_to_entity_context,
    json_to_relation_context,
    json_to_text_unit_context,
)


class TestLightRagContextTypes:
    """Test cases for LightRAG context type conversion functions."""

    def test_json_to_entity_context_basic(self):
        """Test basic entity context conversion."""
        json_data = {
            "id": "1",
            "entity": "章武三年",
            "type": "event",
            "description": "章武三年是刘备去世的年份，标志着蜀汉历史的重要转折点。",
            "rank": 1,
            "created_at": "2025-06-18 23:04:05",
            "file_path": "/var/folders/test/chapter_001.txt",
        }

        result = json_to_entity_context(json_data)

        assert isinstance(result, LightRagEntityContext)
        assert result.id == "1"
        assert result.entity == "章武三年"
        assert result.type == "event"
        assert result.description == "章武三年是刘备去世的年份，标志着蜀汉历史的重要转折点。"
        assert result.rank == 1
        assert result.created_at == datetime(2025, 6, 18, 23, 4, 5)
        assert result.file_path == ["/var/folders/test/chapter_001.txt"]

    def test_json_to_entity_context_with_sep_file_path(self):
        """Test entity context conversion with SEP-separated file paths."""
        json_data = {
            "id": "2",
            "entity": "诸葛亮",
            "type": "person",
            "description": "蜀汉丞相",
            "rank": 5,
            "created_at": "2025-06-18 22:00:00",
            "file_path": "/var/folders/test/chapter_001.txt<SEP>/var/folders/test/chapter_002.txt<SEP>/var/folders/test/chapter_003.txt",
        }

        result = json_to_entity_context(json_data)

        assert result.file_path == [
            "/var/folders/test/chapter_001.txt",
            "/var/folders/test/chapter_002.txt",
            "/var/folders/test/chapter_003.txt",
        ]

    def test_json_to_entity_context_optional_fields(self):
        """Test entity context conversion with optional fields missing."""
        json_data = {
            "id": "3",
            "entity": "曹操",
            "type": "person",
            "created_at": "2025-06-18 20:00:00",
        }

        result = json_to_entity_context(json_data)

        assert result.id == "3"
        assert result.entity == "曹操"
        assert result.type == "person"
        assert result.description is None
        assert result.rank is None
        assert result.file_path is None
        assert result.created_at == datetime(2025, 6, 18, 20, 0, 0)

    def test_json_to_relation_context_basic(self):
        """Test basic relation context conversion."""
        json_data = {
            "id": "1",
            "entity1": "关公之死",
            "entity2": "孙权",
            "description": "关公之死是孙权获得胜利的重要事件，标志着东吴的胜利。",
            "keywords": "历史事件,胜利",
            "weight": 9.0,
            "rank": 59,
            "created_at": "2025-06-18 22:33:17",
            "file_path": "/var/folders/test/chapter_077.txt",
        }

        result = json_to_relation_context(json_data)

        assert isinstance(result, LightRagRelationContext)
        assert result.id == "1"
        assert result.entity1 == "关公之死"
        assert result.entity2 == "孙权"
        assert result.description == "关公之死是孙权获得胜利的重要事件，标志着东吴的胜利。"
        assert result.keywords == "历史事件,胜利"
        assert result.weight == 9.0
        assert result.rank == 59
        assert result.created_at == datetime(2025, 6, 18, 22, 33, 17)
        assert result.file_path == ["/var/folders/test/chapter_077.txt"]

    def test_json_to_relation_context_complex_file_path(self):
        """Test relation context conversion with complex file path and description."""
        json_data = {
            "id": "2",
            "entity1": "丧事",
            "entity2": "孙策",
            "description": "孙策的去世引发了丧事的举行，影响了江东的政治局势。<SEP>孙策的去世引发了丧事的举行，涉及东吴的文武官员。",
            "keywords": "丧失,事件,仪式,政治影响",
            "weight": 16.0,
            "rank": 34,
            "created_at": "2025-06-18 18:23:49",
            "file_path": "/var/folders/test/chapter_029a.txt<SEP>/var/folders/test/chapter_029b.txt",
        }

        result = json_to_relation_context(json_data)

        assert (
            result.description
            == "孙策的去世引发了丧事的举行，影响了江东的政治局势。<SEP>孙策的去世引发了丧事的举行，涉及东吴的文武官员。"
        )
        assert result.file_path == [
            "/var/folders/test/chapter_029a.txt",
            "/var/folders/test/chapter_029b.txt",
        ]

    def test_json_to_text_unit_context_basic(self):
        """Test basic text unit context conversion."""
        json_data = {
            "id": "1",
            "content": "《三国演义》第八十五回 刘先主遗诏托孤儿 诸葛亮安居平五路",
            "file_path": "/var/folders/test/chapter_085.txt",
        }

        result = json_to_text_unit_context(json_data)

        assert isinstance(result, LightRagTextUnitContext)
        assert result.id == "1"
        assert result.content == "《三国演义》第八十五回 刘先主遗诏托孤儿 诸葛亮安居平五路"
        assert result.file_path == ["/var/folders/test/chapter_085.txt"]

    def test_json_to_text_unit_context_no_file_path(self):
        """Test text unit context conversion without file path."""
        json_data = {
            "id": "2",
            "content": "这是一段测试文本内容。",
        }

        result = json_to_text_unit_context(json_data)

        assert result.id == "2"
        assert result.content == "这是一段测试文本内容。"
        assert result.file_path is None

    def test_json_list_to_relation_contexts_with_real_data(self):
        """Test relation context list conversion with real data provided by user."""
        json_list = [
            {
                "id": "1",
                "entity1": "关公之死",
                "entity2": "孙权",
                "description": "关公之死是孙权获得胜利的重要事件，标志着东吴的胜利。",
                "keywords": "历史事件,胜利",
                "weight": 9.0,
                "rank": 59,
                "created_at": "2025-06-18 22:33:17",
                "file_path": "/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_077bgmy3rl7.txt",
            },
            {
                "id": "2",
                "entity1": "丧事",
                "entity2": "孙策",
                "description": "孙策的去世引发了丧事的举行，影响了江东的政治局势。<SEP>孙策的去世引发了丧事的举行，涉及东吴的文武官员。",
                "keywords": "丧失,事件,仪式,政治影响",
                "weight": 16.0,
                "rank": 34,
                "created_at": "2025-06-18 18:23:49",
                "file_path": "/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_029v59r9fpr.txt<SEP>/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_0294zl9gdlm.txt",
            },
            {
                "id": "3",
                "entity1": "董卓之死",
                "entity2": "长安",
                "description": "董卓之死发生在长安，标志着权力斗争的高潮。",
                "keywords": "事件地点,权力斗争",
                "weight": 8.0,
                "rank": 32,
                "created_at": "2025-06-18 17:26:21",
                "file_path": "/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_009sr1ffq70.txt",
            },
        ]

        results = json_list_to_relation_contexts(json_list)

        assert len(results) == 3
        assert all(isinstance(r, LightRagRelationContext) for r in results)

        # Test first relation
        assert results[0].id == "1"
        assert results[0].entity1 == "关公之死"
        assert results[0].entity2 == "孙权"
        assert results[0].weight == 9.0

        # Test second relation with complex file path
        assert results[1].id == "2"
        assert len(results[1].file_path) == 2
        assert "chapter_029v59r9fpr.txt" in results[1].file_path[0]
        assert "chapter_0294zl9gdlm.txt" in results[1].file_path[1]

        # Test third relation
        assert results[2].entity1 == "董卓之死"
        assert results[2].entity2 == "长安"

    def test_json_list_to_entity_contexts_empty_list(self):
        """Test entity context list conversion with empty list."""
        result = json_list_to_entity_contexts([])
        assert result == []

    def test_json_list_to_relation_contexts_empty_list(self):
        """Test relation context list conversion with empty list."""
        result = json_list_to_relation_contexts([])
        assert result == []

    def test_json_list_to_text_unit_contexts_empty_list(self):
        """Test text unit context list conversion with empty list."""
        result = json_list_to_text_unit_contexts([])
        assert result == []

    def test_file_path_as_list_input(self):
        """Test conversion when file_path is already a list."""
        json_data = {
            "id": "1",
            "entity": "测试实体",
            "type": "test",
            "created_at": "2025-06-18 12:00:00",
            "file_path": ["/path1.txt", "/path2.txt", "/path3.txt"],
        }

        result = json_to_entity_context(json_data)

        assert result.file_path == ["/path1.txt", "/path2.txt", "/path3.txt"]

    def test_complex_relation_data_with_long_description(self):
        """Test relation conversion with very long description containing SEP."""
        json_data = {
            "id": "6",
            "entity1": "吕翔",
            "entity2": "曹操",
            "description": "吕翔作为曹操的将领，参与了对袁谭的战争，并因战功被封为列侯。<SEP>吕翔是曹操的重要将领之一，他与吕旷一起被曹操封为列侯。在曹操的指挥下，吕翔参与了多场战争，包括对冀州的攻打和对袁氏家族的战争。在这些军事行动中，吕翔表现出色，因其卓越的战功获得了封赏。他支持曹操的军事战略，尤其是在对袁谭的战争中，展现了他的忠诚与能力。总的来说，吕翔在曹操的统治下，凭借其出色的军事表现，赢得了列侯的荣誉。",
            "keywords": "军事关系,军事合作,军事指挥,军事支持,将领,将领关系,战争,战功,部将,部将关系",
            "weight": 53.0,
            "rank": 264,
            "created_at": "2025-06-18 23:32:42",
            "file_path": "/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_032cmn9iwo_.txt<SEP>/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_032n_bihr49.txt<SEP>/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_03260rehl_x.txt<SEP>/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_032r9oc87ku.txt<SEP>/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_032ctna0u81.txt<SEP>/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_032z0cv07_b.txt<SEP>/var/folders/rk/5c6v7x7513lbk7svs7pf4kr00000gn/T/chapter_032ee4c4_i9.txt",
        }

        result = json_to_relation_context(json_data)

        assert result.entity1 == "吕翔"
        assert result.entity2 == "曹操"
        assert result.weight == 53.0
        assert result.rank == 264
        assert len(result.file_path) == 7  # 7 files separated by <SEP>
        assert "chapter_032cmn9iwo_.txt" in result.file_path[0]
        assert "chapter_032ee4c4_i9.txt" in result.file_path[6]
        # Verify description contains SEP and is preserved as-is
        assert "<SEP>" in result.description

    def test_datetime_parsing_edge_cases(self):
        """Test datetime parsing with different time formats."""
        json_data = {
            "id": "1",
            "entity": "测试",
            "type": "test",
            "created_at": "2025-12-31 23:59:59",  # End of year
        }

        result = json_to_entity_context(json_data)
        assert result.created_at == datetime(2025, 12, 31, 23, 59, 59)

    def test_error_handling_invalid_datetime(self):
        """Test error handling for invalid datetime format."""
        json_data = {
            "id": "1",
            "entity": "测试",
            "type": "test",
            "created_at": "invalid-date-format",
        }

        with pytest.raises(ValueError):
            json_to_entity_context(json_data)

    def test_error_handling_missing_required_fields(self):
        """Test error handling for missing required fields."""
        # Missing 'entity' field for entity context
        json_data = {
            "id": "1",
            "type": "test",
            "created_at": "2025-06-18 12:00:00",
        }

        with pytest.raises(KeyError):
            json_to_entity_context(json_data)

        # Missing 'content' field for text unit context
        json_data = {
            "id": "1",
        }

        with pytest.raises(KeyError):
            json_to_text_unit_context(json_data)

    def test_batch_conversion_performance(self):
        """Test batch conversion with large dataset."""
        # Create a large list of test data
        large_json_list = []
        for i in range(100):
            large_json_list.append(
                {
                    "id": str(i),
                    "entity1": f"实体{i}A",
                    "entity2": f"实体{i}B",
                    "description": f"这是第{i}个关系的描述",
                    "keywords": f"关键词{i},测试",
                    "weight": float(i),
                    "rank": i,
                    "created_at": "2025-06-18 12:00:00",
                    "file_path": f"/test/file_{i}.txt",
                }
            )

        results = json_list_to_relation_contexts(large_json_list)

        assert len(results) == 100
        assert results[0].entity1 == "实体0A"
        assert results[99].entity1 == "实体99A"
        assert results[50].weight == 50.0


if __name__ == "__main__":
    pytest.main([__file__])
