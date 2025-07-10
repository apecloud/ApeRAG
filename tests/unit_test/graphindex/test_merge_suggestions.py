"""
Unit tests for merge suggestions pure functions in LightRAG.
Tests parse_single_merge_record, filter_and_deduplicate_suggestions, and related functions.
"""

from unittest.mock import Mock

import pytest

from aperag.graph.lightrag.operate import (
    filter_and_deduplicate_suggestions,
    parse_llm_merge_response,
    parse_single_merge_record,
)
from aperag.graph.lightrag.prompt import GRAPH_FIELD_SEP, PROMPTS
from aperag.graph.lightrag.types import GraphNodeData, MergeSuggestion


class TestParseSingleMergeRecord:
    """Test parse_single_merge_record function."""

    def setup_method(self):
        """Set up test data for each test."""
        # Create test entities
        self.entity1 = GraphNodeData(
            entity_id="中国生态农业学报",
            entity_name="中国生态农业学报",
            entity_type="ORGANIZATION",
            description="中国生态农业学报是一份学术期刊，发表关于生态农业的研究文章",
            degree=8,
        )

        self.entity2 = GraphNodeData(
            entity_id="Chinese Journal of Eco-Agriculture",
            entity_name="Chinese Journal of Eco-Agriculture",
            entity_type="ORGANIZATION",
            description="An academic journal publishing research articles on ecological agriculture",
            degree=6,
        )

        self.entity3 = GraphNodeData(
            entity_id="Apple Inc",
            entity_name="Apple Inc",
            entity_type="ORGANIZATION",
            description="Apple Inc. is an American multinational technology company",
            degree=15,
        )

        self.entity4 = GraphNodeData(
            entity_id="Apple",
            entity_name="Apple",
            entity_type="ORGANIZATION",
            description="Technology company known for iPhone and Mac products",
            degree=12,
        )

        # Add test entity for duplicate handling test
        self.automotive_sales_entity = GraphNodeData(
            entity_id="Automotive Sales",
            entity_name="Automotive Sales",
            entity_type="category",
            description="Automotive Sales is a major source of revenue for Tesla, Inc., representing the income generated from the sale of vehicles.",
            degree=1,
        )

        # Create entity lookup dictionary
        self.entity_lookup = {
            "中国生态农业学报": self.entity1,
            "Chinese Journal of Eco-Agriculture": self.entity2,
            "Apple Inc": self.entity3,
            "Apple": self.entity4,
            "Automotive Sales": self.automotive_sales_entity,
        }

        # Create mock logger
        self.mock_logger = Mock()

    def test_successful_parsing_chinese_journals(self):
        """Test successful parsing of the Chinese journal example."""
        record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}中国生态农业学报{GRAPH_FIELD_SEP}Chinese Journal of Eco-Agriculture{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.95{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}These entities are the Chinese and English names for the same academic journal{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}中国生态农业学报{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION)'

        result = parse_single_merge_record(
            record, self.entity_lookup, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is not None
        assert len(result.entities) == 2
        assert result.entities[0].entity_name == "中国生态农业学报"
        assert result.entities[1].entity_name == "Chinese Journal of Eco-Agriculture"
        assert result.confidence_score == 0.95
        assert "Chinese and English names" in result.merge_reason
        # Check that suggested_target_entity is now a GraphNodeData object
        assert isinstance(result.suggested_target_entity, GraphNodeData)
        assert result.suggested_target_entity.entity_name == "中国生态农业学报"
        assert result.suggested_target_entity.entity_type == "ORGANIZATION"

    def test_duplicate_entity_names_handling(self):
        """Test that duplicate entity names in LLM response are properly handled."""
        from aperag.graph.lightrag.prompt import GRAPH_FIELD_SEP, PROMPTS

        # Create a record with duplicate entity names (this is the bug scenario)
        duplicate_entity_record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Automotive Sales{GRAPH_FIELD_SEP}Automotive Sales{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}1.0{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}These are identical entities and should be merged to remove redundancy.{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Automotive Sales{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}category)'

        result = parse_single_merge_record(
            duplicate_entity_record, self.entity_lookup, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        # Should return None because after deduplication, we don't have enough unique entities
        assert result is None

        # Verify that the debug message was logged
        self.mock_logger.debug.assert_any_call("Skipping duplicate entity name 'Automotive Sales' in merge suggestion")
        self.mock_logger.debug.assert_any_call("Not enough unique valid entities found: ['Automotive Sales']")

    def test_successful_parsing_with_quotes(self):
        """Test parsing when description has quotes."""
        record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.92{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Both entities refer to the same technology company{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION)'

        result = parse_single_merge_record(
            record, self.entity_lookup, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is not None
        assert len(result.entities) == 2
        assert result.confidence_score == 0.92
        # The suggested_target_entity no longer includes description
        assert result.suggested_target_entity.entity_name == "Apple Inc"
        assert result.suggested_target_entity.entity_type == "ORGANIZATION"

    def test_confidence_below_threshold(self):
        """Test rejection of suggestions below confidence threshold."""
        record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.4{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Low confidence suggestion{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION)'

        result = parse_single_merge_record(
            record, self.entity_lookup, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is None
        self.mock_logger.debug.assert_called()

    def test_invalid_confidence_score(self):
        """Test handling of invalid confidence score."""
        record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}invalid_score{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Reason{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION)'

        result = parse_single_merge_record(
            record, self.entity_lookup, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is None
        self.mock_logger.warning.assert_called()

    def test_insufficient_parts(self):
        """Test handling of records with insufficient parts."""
        record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.9{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Reason{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc)'  # Missing parts

        result = parse_single_merge_record(
            record, self.entity_lookup, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is None
        self.mock_logger.warning.assert_called()

    def test_entity_not_in_lookup(self):
        """Test handling when entity names are not found in lookup."""
        record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Unknown Entity{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.9{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Reason{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION)'

        result = parse_single_merge_record(
            record, self.entity_lookup, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is None
        self.mock_logger.debug.assert_called()

    def test_insufficient_valid_entities(self):
        """Test handling when fewer than 2 valid entities are found."""
        record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.9{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Reason{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION)'  # Only one entity

        result = parse_single_merge_record(
            record, self.entity_lookup, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is None

    def test_malformed_record_format(self):
        """Test handling of malformed record format."""
        record = "invalid_format_without_merge_group"

        result = parse_single_merge_record(
            record, self.entity_lookup, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is None
        self.mock_logger.warning.assert_called()

    def test_entity_names_with_commas(self):
        """Test that entity names containing commas are handled correctly with GRAPH_FIELD_SEP."""
        # Add entity with comma in name
        entity_with_comma = GraphNodeData(
            entity_id="Company, Inc.",
            entity_name="Company, Inc.",
            entity_type="ORGANIZATION",
            description="A company with comma in name",
            degree=8,
        )

        lookup_with_comma = {**self.entity_lookup, "Company, Inc.": entity_with_comma}

        record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Company, Inc.{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.8{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Reason{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Company, Inc.{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION)'

        result = parse_single_merge_record(
            record, lookup_with_comma, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is not None
        assert len(result.entities) == 2
        assert result.entities[0].entity_name == "Company, Inc."

    def test_three_entity_merge(self):
        """Test parsing suggestion with three entities."""
        # Add a third Apple entity
        entity5 = GraphNodeData(
            entity_id="Apple Corporation",
            entity_name="Apple Corporation",
            entity_type="ORGANIZATION",
            description="Another Apple entity",
            degree=10,
        )

        lookup_three = {**self.entity_lookup, "Apple Corporation": entity5}

        record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{GRAPH_FIELD_SEP}Apple{GRAPH_FIELD_SEP}Apple Corporation{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.88{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}All refer to same company{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION)'

        result = parse_single_merge_record(
            record, lookup_three, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is not None
        assert len(result.entities) == 3

    def test_empty_parts_filtering(self):
        """Test that empty parts are properly filtered out."""
        # This simulates the original issue where content starts with delimiter
        record = f'("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.92{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Reason{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION)'

        result = parse_single_merge_record(
            record, self.entity_lookup, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert result is not None
        assert len(result.entities) == 2


class TestFilterAndDeduplicateSuggestions:
    """Test filter_and_deduplicate_suggestions function."""

    def setup_method(self):
        """Set up test data for each test."""
        # Create mock entities
        entity1 = GraphNodeData(entity_id="A", entity_name="A", entity_type="ORG", description="Entity A", degree=5)
        entity2 = GraphNodeData(entity_id="B", entity_name="B", entity_type="ORG", description="Entity B", degree=3)
        entity3 = GraphNodeData(entity_id="C", entity_name="C", entity_type="ORG", description="Entity C", degree=7)
        entity4 = GraphNodeData(entity_id="D", entity_name="D", entity_type="ORG", description="Entity D", degree=2)

        # Create target entity data as GraphNodeData objects for MergeSuggestion
        target_entity_a = GraphNodeData(
            entity_id="A",
            entity_name="A",
            entity_type="ORG",
            description="Merged entity A-B",
        )

        target_entity_c = GraphNodeData(
            entity_id="C",
            entity_name="C",
            entity_type="ORG",
            description="Merged entity C-D",
        )

        target_entity_b = GraphNodeData(
            entity_id="B",
            entity_name="B",
            entity_type="ORG",
            description="Merged entity B-A",
        )

        # Create test suggestions with different confidence scores
        self.suggestion1 = MergeSuggestion(
            entities=[entity1, entity2],
            confidence_score=0.95,
            merge_reason="High confidence merge",
            suggested_target_entity=target_entity_a,
        )

        self.suggestion2 = MergeSuggestion(
            entities=[entity3, entity4],
            confidence_score=0.87,
            merge_reason="Medium confidence merge",
            suggested_target_entity=target_entity_c,
        )

        # Use completely different entities to avoid overlap in deduplication tests
        entity5 = GraphNodeData(entity_id="E", entity_name="E", entity_type="ORG", description="Entity E", degree=4)
        entity6 = GraphNodeData(entity_id="F", entity_name="F", entity_type="ORG", description="Entity F", degree=6)
        target_entity_e = GraphNodeData(
            entity_id="E",
            entity_name="E",
            entity_type="ORG",
            description="Merged entity E-F",
        )

        self.suggestion3 = MergeSuggestion(
            entities=[entity5, entity6],  # Completely different pair to avoid any overlap
            confidence_score=0.92,
            merge_reason="Another high confidence merge",
            suggested_target_entity=target_entity_e,
        )

        # Duplicate suggestion (same entities, different order)
        self.suggestion4 = MergeSuggestion(
            entities=[entity2, entity1],  # Same as suggestion1 but reversed
            confidence_score=0.89,
            merge_reason="Duplicate merge suggestion",
            suggested_target_entity=target_entity_b,
        )

    def test_basic_filtering_and_sorting(self):
        """Test basic filtering and sorting by confidence score."""
        suggestions = [self.suggestion2, self.suggestion1, self.suggestion3]  # Unsorted
        max_suggestions = 10

        result = filter_and_deduplicate_suggestions(suggestions, max_suggestions)

        # Should be sorted by confidence score (highest first)
        assert len(result) == 3
        assert result[0].confidence_score == 0.95
        assert result[1].confidence_score == 0.92
        assert result[2].confidence_score == 0.87

    def test_max_suggestions_limit(self):
        """Test that max_suggestions limit is respected."""
        suggestions = [self.suggestion1, self.suggestion2, self.suggestion3]
        max_suggestions = 2

        result = filter_and_deduplicate_suggestions(suggestions, max_suggestions)

        assert len(result) == 2
        assert result[0].confidence_score == 0.95  # Highest
        assert result[1].confidence_score == 0.92  # Second highest

    def test_deduplication(self):
        """Test deduplication of suggestions with same entity sets."""
        suggestions = [
            self.suggestion1,
            self.suggestion4,
            self.suggestion2,
        ]  # suggestion1 and suggestion4 are duplicates
        max_suggestions = 10

        result = filter_and_deduplicate_suggestions(suggestions, max_suggestions)

        # Should keep only one of the duplicate suggestions (the one with higher confidence)
        assert len(result) == 2
        assert result[0].confidence_score == 0.95  # suggestion1 (higher confidence)
        assert result[1].confidence_score == 0.87  # suggestion2

    def test_empty_suggestions_list(self):
        """Test handling of empty suggestions list."""
        suggestions = []
        max_suggestions = 10

        result = filter_and_deduplicate_suggestions(suggestions, max_suggestions)

        assert len(result) == 0

    def test_zero_max_suggestions(self):
        """Test handling of zero max_suggestions."""
        suggestions = [self.suggestion1, self.suggestion2]
        max_suggestions = 0

        result = filter_and_deduplicate_suggestions(suggestions, max_suggestions)

        assert len(result) == 0

    def test_single_suggestion(self):
        """Test handling of single suggestion."""
        suggestions = [self.suggestion1]
        max_suggestions = 10

        result = filter_and_deduplicate_suggestions(suggestions, max_suggestions)

        assert len(result) == 1
        assert result[0] == self.suggestion1


class TestParseLlmMergeResponse:
    """Test parse_llm_merge_response function."""

    def setup_method(self):
        """Set up test data for each test."""
        # Create test entities
        self.entity1 = GraphNodeData(
            entity_id="Apple Inc",
            entity_name="Apple Inc",
            entity_type="ORGANIZATION",
            description="Apple Inc. is an American multinational technology company",
            degree=15,
        )

        self.entity2 = GraphNodeData(
            entity_id="Apple",
            entity_name="Apple",
            entity_type="ORGANIZATION",
            description="Technology company known for iPhone and Mac products",
            degree=12,
        )

        self.entity3 = GraphNodeData(
            entity_id="Microsoft",
            entity_name="Microsoft",
            entity_type="ORGANIZATION",
            description="Software company",
            degree=10,
        )

        # Create entities list for the function
        self.entities_list = [self.entity1, self.entity2, self.entity3]

        # Create mock logger
        self.mock_logger = Mock()

    def test_successful_parsing_multiple_records(self):
        """Test successful parsing of multiple merge records."""
        llm_response = f"""Here are the merge suggestions:

("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.92{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Both refer to same company{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION){PROMPTS["DEFAULT_RECORD_DELIMITER"]}
("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Microsoft{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.15{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Different companies{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Microsoft{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION){PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}"""

        result = parse_llm_merge_response(
            llm_response, self.entities_list, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        # Should only include the first suggestion (confidence 0.92 > 0.6)
        assert len(result) == 1
        assert result[0].confidence_score == 0.92
        assert len(result[0].entities) == 2

    def test_empty_response(self):
        """Test handling of empty LLM response."""
        llm_response = ""

        result = parse_llm_merge_response(
            llm_response, self.entities_list, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert len(result) == 0

    def test_no_merge_records(self):
        """Test handling of response with no merge records."""
        llm_response = f"No merge suggestions found.{PROMPTS['DEFAULT_COMPLETION_DELIMITER']}"

        result = parse_llm_merge_response(
            llm_response, self.entities_list, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert len(result) == 0

    def test_malformed_records_filtered_out(self):
        """Test that malformed records are filtered out."""
        llm_response = f"""Here are the suggestions:

("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.92{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Good record{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION){PROMPTS["DEFAULT_RECORD_DELIMITER"]}
malformed_record_without_proper_format{PROMPTS["DEFAULT_RECORD_DELIMITER"]}
("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Microsoft{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}incomplete_record){PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}"""

        result = parse_llm_merge_response(
            llm_response, self.entities_list, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        # Should only include the valid record
        assert len(result) == 1
        assert result[0].confidence_score == 0.92

    def test_confidence_threshold_filtering(self):
        """Test that suggestions below confidence threshold are filtered out."""
        llm_response = f"""Suggestions:

("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.75{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}High confidence{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION){PROMPTS["DEFAULT_RECORD_DELIMITER"]}
("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Microsoft{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.45{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Low confidence{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Microsoft{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION){PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}"""

        result = parse_llm_merge_response(
            llm_response, self.entities_list, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        # Should only include the high confidence suggestion
        assert len(result) == 1
        assert result[0].confidence_score == 0.75

    def test_entity_lookup_creation(self):
        """Test that entity lookup is correctly created from entities_list."""
        llm_response = f"""Here is the merge suggestion:

("merge_group"{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{GRAPH_FIELD_SEP}Apple{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}0.85{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Same company{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}Apple Inc{PROMPTS["DEFAULT_TUPLE_DELIMITER"]}ORGANIZATION){PROMPTS["DEFAULT_RECORD_DELIMITER"]}
{PROMPTS["DEFAULT_COMPLETION_DELIMITER"]}"""

        result = parse_llm_merge_response(
            llm_response, self.entities_list, confidence_threshold=0.6, lightrag_logger=self.mock_logger
        )

        assert len(result) == 1
        # Verify that the entities were correctly looked up
        assert result[0].entities[0].entity_name == "Apple Inc"
        assert result[0].entities[1].entity_name == "Apple"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
