"""
Unit tests for GraphService
"""

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from aperag.db.models import MergeSuggestionStatus
from aperag.service.graph_service import GraphService


class TestGraphService:
    def setup_method(self):
        """Setup test fixtures"""
        self.graph_service = GraphService()

    @pytest.mark.asyncio
    async def test_merge_nodes_direct(self):
        """Test direct merge using entity_ids"""
        # Mock dependencies
        mock_collection = MagicMock()
        mock_rag = AsyncMock()

        with (
            patch.object(self.graph_service, "_get_and_validate_collection", return_value=mock_collection),
            patch("aperag.graph.lightrag_manager.create_lightrag_instance", return_value=mock_rag),
        ):
            # Mock the LightRAG merge operation
            mock_rag.amerge_nodes.return_value = {
                "status": "success",
                "message": "Successfully merged 2 entities into entity1",
                "target_entity_data": {
                    "entity_name": "entity1",
                    "entity_type": "ORGANIZATION",
                    "description": "Merged description",
                },
                "source_entities": ["entity2"],
                "redirected_edges": 5,
                "merged_description_length": 100,
            }

            # Test the merge
            result = await self.graph_service.merge_nodes(
                user_id="user123",
                collection_id="col123",
                entity_ids=["entity1", "entity2"],
                target_entity_data={"entity_name": "Custom Name"},
            )

            # Verify the result
            assert result["status"] == "success"
            assert result["entity_ids"] == ["entity1", "entity2"]
            assert "target_entity_data" in result

            # Verify LightRAG was called correctly
            mock_rag.amerge_nodes.assert_called_once_with(
                entity_ids=["entity1", "entity2"], target_entity_data={"entity_name": "Custom Name"}
            )
            mock_rag.finalize_storages.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_suggestion_action_accept(self):
        """Test accepting a merge suggestion"""
        # Mock dependencies
        mock_collection = MagicMock()
        mock_rag = AsyncMock()
        mock_suggestion = MagicMock()
        mock_suggestion.id = "sug123"
        mock_suggestion.collection_id = "col123"
        mock_suggestion.status = MergeSuggestionStatus.PENDING
        mock_suggestion.entity_ids = ["entity1", "entity2"]
        mock_suggestion.suggested_target_entity = {"entity_name": "entity1"}

        with (
            patch.object(self.graph_service, "_get_and_validate_collection", return_value=mock_collection),
            patch.object(self.graph_service.db_ops, "get_suggestions_by_ids", return_value=[mock_suggestion]),
            patch.object(self.graph_service.db_ops, "update_suggestion_status"),
            patch.object(self.graph_service.db_ops, "expire_related_suggestions"),
            patch("aperag.graph.lightrag_manager.create_lightrag_instance", return_value=mock_rag),
        ):
            # Mock the LightRAG merge operation
            mock_rag.amerge_nodes.return_value = {
                "status": "success",
                "message": "Successfully merged 2 entities into entity1",
                "target_entity_data": {
                    "entity_name": "entity1",
                    "entity_type": "ORGANIZATION",
                    "description": "Merged description",
                },
                "source_entities": ["entity2"],
                "redirected_edges": 5,
                "merged_description_length": 100,
            }

            # Test accepting the suggestion
            result = await self.graph_service.handle_suggestion_action(
                user_id="user123", collection_id="col123", suggestion_id="sug123", action="accept"
            )

            # Verify the result
            assert result["status"] == "success"
            assert result["suggestion_id"] == "sug123"
            assert result["action"] == "accept"
            assert result["merge_result"] is not None
            assert "has been accepted and merge completed" in result["message"]

            # Verify database operations
            self.graph_service.db_ops.update_suggestion_status.assert_called_once_with(
                "sug123", MergeSuggestionStatus.ACCEPTED, ANY
            )
            self.graph_service.db_ops.expire_related_suggestions.assert_called_once_with(
                "col123", ["entity1", "entity2"]
            )

    @pytest.mark.asyncio
    async def test_handle_suggestion_action_reject(self):
        """Test rejecting a merge suggestion"""
        # Mock dependencies
        mock_collection = MagicMock()
        mock_suggestion = MagicMock()
        mock_suggestion.id = "sug123"
        mock_suggestion.collection_id = "col123"
        mock_suggestion.status = MergeSuggestionStatus.PENDING
        mock_suggestion.entity_ids = ["entity1", "entity2"]

        with (
            patch.object(self.graph_service, "_get_and_validate_collection", return_value=mock_collection),
            patch.object(self.graph_service.db_ops, "get_suggestions_by_ids", return_value=[mock_suggestion]),
            patch.object(self.graph_service.db_ops, "update_suggestion_status"),
        ):
            # Test rejecting the suggestion
            result = await self.graph_service.handle_suggestion_action(
                user_id="user123", collection_id="col123", suggestion_id="sug123", action="reject"
            )

            # Verify the result
            assert result["status"] == "success"
            assert result["suggestion_id"] == "sug123"
            assert result["action"] == "reject"
            assert result["merge_result"] is None
            assert "has been rejected" in result["message"]

            # Verify database operations
            self.graph_service.db_ops.update_suggestion_status.assert_called_once_with(
                "sug123", MergeSuggestionStatus.REJECTED, ANY
            )

    @pytest.mark.asyncio
    async def test_handle_suggestion_action_invalid_action(self):
        """Test invalid action parameter"""
        mock_collection = MagicMock()

        with patch.object(self.graph_service, "_get_and_validate_collection", return_value=mock_collection):
            with pytest.raises(ValueError, match="Invalid action: invalid"):
                await self.graph_service.handle_suggestion_action(
                    user_id="user123", collection_id="col123", suggestion_id="sug123", action="invalid"
                )

    @pytest.mark.asyncio
    async def test_handle_suggestion_action_suggestion_not_found(self):
        """Test suggestion not found error"""
        mock_collection = MagicMock()

        with (
            patch.object(self.graph_service, "_get_and_validate_collection", return_value=mock_collection),
            patch.object(self.graph_service.db_ops, "get_suggestions_by_ids", return_value=[]),
        ):
            with pytest.raises(ValueError, match="Suggestion not found: sug123"):
                await self.graph_service.handle_suggestion_action(
                    user_id="user123", collection_id="col123", suggestion_id="sug123", action="accept"
                )

    @pytest.mark.asyncio
    async def test_handle_suggestion_action_non_pending_status(self):
        """Test suggestion with non-pending status error"""
        mock_collection = MagicMock()
        mock_suggestion = MagicMock()
        mock_suggestion.status = MergeSuggestionStatus.ACCEPTED

        with (
            patch.object(self.graph_service, "_get_and_validate_collection", return_value=mock_collection),
            patch.object(self.graph_service.db_ops, "get_suggestions_by_ids", return_value=[mock_suggestion]),
        ):
            with pytest.raises(ValueError, match="Cannot act on suggestion with status"):
                await self.graph_service.handle_suggestion_action(
                    user_id="user123", collection_id="col123", suggestion_id="sug123", action="accept"
                )

    @pytest.mark.asyncio
    async def test_merge_nodes_empty_entity_ids(self):
        """Test merge with empty entity_ids"""
        with pytest.raises(ValueError, match="entity_ids cannot be empty"):
            await self.graph_service.merge_nodes(user_id="user123", collection_id="col123", entity_ids=[])
