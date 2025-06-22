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

import json
import time
from http import HTTPStatus
from typing import Dict, Any, Optional

import pytest

from tests.e2e_test.config import (
    COMPLETION_MODEL_CUSTOM_PROVIDER,
    COMPLETION_MODEL_NAME,
    COMPLETION_MODEL_PROVIDER,
    EMBEDDING_MODEL_CUSTOM_PROVIDER,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PROVIDER,
)


class AuditLogTestHelper:
    """Helper class for audit log testing"""
    
    def __init__(self, cookie_client):
        self.cookie_client = cookie_client
    
    def get_audit_logs(self, **filters) -> list:
        """Get audit logs with optional filters"""
        params = {k: v for k, v in filters.items() if v is not None}
        resp = self.cookie_client.get("/api/v1/audit-logs", params=params)
        assert resp.status_code == HTTPStatus.OK, f"Failed to get audit logs: {resp.text}"
        return resp.json()["items"]
    
    def find_audit_log(self, resource_type: str, api_name: str, resource_id: Optional[str] = None,
                      http_method: str = None, max_wait_seconds: int = 10, 
                      match_response_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Find a specific audit log by criteria with retry mechanism"""
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            logs = self.get_audit_logs(
                resource_type=resource_type,
                api_name=api_name,
                http_method=http_method
            )
            
            for log in logs:
                # Check resource_id if provided and available
                if resource_id and log.get("resource_id") and log.get("resource_id") != resource_id:
                    continue
                if http_method and log.get("http_method") != http_method:
                    continue
                
                # For create operations, match against response data when resource_id is not available in path
                if (resource_id and not log.get("resource_id") and 
                    match_response_data and log.get("response_data")):
                    try:
                        response_data = json.loads(log.get("response_data", "{}"))
                        match_found = True
                        for key, expected_value in match_response_data.items():
                            if response_data.get(key) != expected_value:
                                match_found = False
                                break
                        if not match_found:
                            continue
                    except (json.JSONDecodeError, AttributeError):
                        continue
                
                return log
            
            time.sleep(0.5)  # Wait before retry
        
        return None
    
    def assert_audit_log_content(self, log: Dict[str, Any], expected_fields: Dict[str, Any]):
        """Assert audit log contains expected fields and values"""
        assert log is not None, "Audit log not found"
        
        for field, expected_value in expected_fields.items():
            actual_value = log.get(field)
            if expected_value is not None:
                assert actual_value == expected_value, f"Field {field}: expected {expected_value}, got {actual_value}"
            else:
                assert field in log, f"Field {field} not found in audit log"
        
        # Basic assertions for all audit logs
        assert log.get("start_time") is not None, "start_time should not be None"
        assert log.get("end_time") is not None, "end_time should not be None"
        assert log.get("duration_ms") is not None, "duration_ms should not be None"
        assert log.get("status_code") in [200, 201, 204], f"Unexpected status code: {log.get('status_code')}"
        assert log.get("user_id") is not None, "user_id should not be None"
        assert log.get("username") is not None, "username should not be None"
    
    def parse_json_field(self, log: Dict[str, Any], field_name: str) -> Dict[str, Any]:
        """Parse JSON string field from audit log"""
        # Handle both dict and object access
        if hasattr(log, field_name):
            json_str = getattr(log, field_name, "{}")
        else:
            json_str = log.get(field_name, "{}")
            
        if json_str:
            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}


@pytest.fixture
def audit_helper(cookie_client):
    """Create audit log test helper"""
    return AuditLogTestHelper(cookie_client)


class TestCollectionAudit:
    """Test audit logs for collection operations"""
    
    def test_create_collection_audit(self, client, audit_helper):
        """Test that creating a collection generates audit log"""
        # Create collection
        collection_data = {
            "title": "Audit Test Collection",
            "type": "document",
            "config": {
                "source": "system",
                "enable_knowledge_graph": False,
                "embedding": {
                    "model": EMBEDDING_MODEL_NAME,
                    "model_service_provider": EMBEDDING_MODEL_PROVIDER,
                    "custom_llm_provider": EMBEDDING_MODEL_CUSTOM_PROVIDER,
                },
            },
        }
        
        resp = client.post("/api/v1/collections", json=collection_data)
        assert resp.status_code == HTTPStatus.OK, f"Failed to create collection: {resp.text}"
        collection = resp.json()
        collection_id = collection["id"]
        
        try:
            # Check audit log
            audit_log = audit_helper.find_audit_log(
                resource_type="collection",
                api_name="CreateCollection",
                resource_id=collection_id,
                http_method="POST",
                match_response_data={"id": collection_id}
            )
            
            audit_helper.assert_audit_log_content(audit_log, {
                "resource_type": "collection",
                "api_name": "CreateCollection",
                "http_method": "POST",
                "path": "/api/v1/collections"
            })
            
            # Verify request data contains collection info
            request_data = audit_helper.parse_json_field(audit_log, "request_data")
            assert "title" in request_data, "Request data should contain title"
            assert request_data["title"] == collection_data["title"]
            
            # Verify response data contains collection ID
            response_data = audit_helper.parse_json_field(audit_log, "response_data")
            assert "id" in response_data, "Response data should contain collection ID"
            assert response_data["id"] == collection_id
            
        finally:
            # Cleanup
            client.delete(f"/api/v1/collections/{collection_id}")
    
    def test_update_collection_audit(self, client, audit_helper, collection):
        """Test that updating a collection generates audit log"""
        collection_id = collection["id"]
        
        # Update collection
        update_data = {
            "title": "Updated Audit Test Collection",
            "description": "Updated description for audit test",
            "config": collection["config"]
        }
        
        resp = client.put(f"/api/v1/collections/{collection_id}", json=update_data)
        assert resp.status_code == HTTPStatus.OK, f"Failed to update collection: {resp.text}"
        
        # Check audit log
        audit_log = audit_helper.find_audit_log(
            resource_type="collection",
            api_name="UpdateCollection",
            resource_id=collection_id,
            http_method="PUT"
        )
        
        audit_helper.assert_audit_log_content(audit_log, {
            "resource_type": "collection",
            "api_name": "UpdateCollection", 
            "resource_id": collection_id,
            "http_method": "PUT",
            "path": f"/api/v1/collections/{collection_id}"
        })
        
        # Verify request data contains update info
        request_data = audit_helper.parse_json_field(audit_log, "request_data")
        # The request data structure might be nested, check both formats
        if "title" in request_data:
            assert request_data["title"] == update_data["title"]
        elif "collection" in request_data and "title" in request_data["collection"]:
            assert request_data["collection"]["title"] == update_data["title"]
        else:
            assert False, f"title field not found in request_data: {request_data}"
    
    def test_delete_collection_audit(self, client, audit_helper):
        """Test that deleting a collection generates audit log"""
        # Create collection first
        collection_data = {
            "title": "Collection to Delete",
            "type": "document",
            "config": {
                "source": "system",
                "enable_knowledge_graph": False,
                "embedding": {
                    "model": EMBEDDING_MODEL_NAME,
                    "model_service_provider": EMBEDDING_MODEL_PROVIDER,
                    "custom_llm_provider": EMBEDDING_MODEL_CUSTOM_PROVIDER,
                },
            },
        }
        
        resp = client.post("/api/v1/collections", json=collection_data)
        assert resp.status_code == HTTPStatus.OK
        collection_id = resp.json()["id"]
        
        # Delete collection
        resp = client.delete(f"/api/v1/collections/{collection_id}")
        assert resp.status_code == HTTPStatus.OK, f"Failed to delete collection: {resp.text}"
        
        # Check audit log
        audit_log = audit_helper.find_audit_log(
            resource_type="collection",
            api_name="DeleteCollection",
            resource_id=collection_id,
            http_method="DELETE"
        )
        
        audit_helper.assert_audit_log_content(audit_log, {
            "resource_type": "collection",
            "api_name": "DeleteCollection",
            "resource_id": collection_id,
            "http_method": "DELETE",
            "path": f"/api/v1/collections/{collection_id}"
        })


class TestDocumentAudit:
    """Test audit logs for document operations"""
    
    def test_create_document_audit(self, client, audit_helper, collection):
        """Test that creating documents generates audit log"""
        collection_id = collection["id"]
        
        # Upload document
        files = {"files": ("audit_test.txt", "This is a test document for audit.", "text/plain")}
        resp = client.post(f"/api/v1/collections/{collection_id}/documents", files=files)
        assert resp.status_code == HTTPStatus.OK, f"Failed to create document: {resp.text}"
        
        documents = resp.json()["items"]
        assert len(documents) > 0
        document_id = documents[0]["id"]
        
        try:
            # Check audit log
            audit_log = audit_helper.find_audit_log(
                resource_type="document",
                api_name="CreateDocuments",
                http_method="POST"
            )
            
            audit_helper.assert_audit_log_content(audit_log, {
                "resource_type": "document",
                "api_name": "CreateDocuments",
                "http_method": "POST",
                "path": f"/api/v1/collections/{collection_id}/documents"
            })
            
            # Verify response data contains document info
            response_data = audit_helper.parse_json_field(audit_log, "response_data")
            assert "items" in response_data
            assert len(response_data["items"]) > 0
            
        finally:
            # Cleanup
            client.delete(f"/api/v1/collections/{collection_id}/documents/{document_id}")
    
    def test_delete_document_audit(self, client, audit_helper, collection, document):
        """Test that deleting a document generates audit log"""
        collection_id = collection["id"]
        document_id = document["id"]
        
        # Delete document
        resp = client.delete(f"/api/v1/collections/{collection_id}/documents/{document_id}")
        assert resp.status_code == HTTPStatus.OK, f"Failed to delete document: {resp.text}"
        
        # Check audit log
        audit_log = audit_helper.find_audit_log(
            resource_type="document",
            api_name="DeleteDocument",
            resource_id=document_id,
            http_method="DELETE"
        )
        
        audit_helper.assert_audit_log_content(audit_log, {
            "resource_type": "document",
            "api_name": "DeleteDocument",
            "resource_id": document_id,
            "http_method": "DELETE",
            "path": f"/api/v1/collections/{collection_id}/documents/{document_id}"
        })


class TestBotAudit:
    """Test audit logs for bot operations"""
    
    def test_create_bot_audit(self, client, audit_helper, collection):
        """Test that creating a bot generates audit log"""
        # Create bot
        config = {
            "model_name": COMPLETION_MODEL_NAME,
            "model_service_provider": COMPLETION_MODEL_PROVIDER,
            "llm": {"context_window": 3500, "similarity_score_threshold": 0.5, "similarity_topk": 3, "temperature": 0.1},
        }
        bot_data = {
            "title": "Audit Test Bot",
            "description": "Bot for audit testing",
            "type": "knowledge",
            "config": json.dumps(config),
            "collection_ids": [collection["id"]],
        }
        
        resp = client.post("/api/v1/bots", json=bot_data)
        assert resp.status_code == HTTPStatus.OK, f"Failed to create bot: {resp.text}"
        bot = resp.json()
        bot_id = bot["id"]
        
        try:
            # Check audit log
            audit_log = audit_helper.find_audit_log(
                resource_type="bot",
                api_name="CreateBot",
                resource_id=bot_id,
                http_method="POST",
                match_response_data={"id": bot_id}
            )
            
            audit_helper.assert_audit_log_content(audit_log, {
                "resource_type": "bot",
                "api_name": "CreateBot",
                "http_method": "POST",
                "path": "/api/v1/bots"
            })
            
            # Verify request data contains bot info
            request_data = audit_helper.parse_json_field(audit_log, "request_data")
            assert "title" in request_data
            assert request_data["title"] == bot_data["title"]
            
        finally:
            # Cleanup
            client.delete(f"/api/v1/bots/{bot_id}")
    
    @pytest.mark.skip(reason="Bot update requires valid LLM Provider configuration with API keys which are not available in test environment")
    def test_update_bot_audit(self, client, audit_helper, bot):
        """Test that updating a bot generates audit log"""
        bot_id = bot["id"]
        
        # Update bot - include required fields
        update_data = {
            "title": "Updated Audit Test Bot",
            "description": "Updated bot description",
            "config": bot["config"],  # Include original config to avoid JSON parsing error
            "collection_ids": bot["collection_ids"]  # Include original collection_ids
        }
        
        resp = client.put(f"/api/v1/bots/{bot_id}", json=update_data)
        assert resp.status_code == HTTPStatus.OK, f"Failed to update bot: {resp.text}"
        
        # Check audit log
        audit_log = audit_helper.find_audit_log(
            resource_type="bot",
            api_name="UpdateBot",
            resource_id=bot_id,
            http_method="PUT"
        )
        
        audit_helper.assert_audit_log_content(audit_log, {
            "resource_type": "bot",
            "api_name": "UpdateBot",
            "resource_id": bot_id,
            "http_method": "PUT",
            "path": f"/api/v1/bots/{bot_id}"
        })
        
        # Verify request data contains update info
        request_data = audit_helper.parse_json_field(audit_log, "request_data")
        assert "title" in request_data
        assert request_data["title"] == update_data["title"]
    
    def test_delete_bot_audit(self, client, audit_helper, collection):
        """Test that deleting a bot generates audit log"""
        # Create bot first
        config = {
            "model_name": COMPLETION_MODEL_NAME,
            "model_service_provider": COMPLETION_MODEL_PROVIDER,
            "llm": {"context_window": 3500, "similarity_score_threshold": 0.5, "similarity_topk": 3, "temperature": 0.1},
        }
        bot_data = {
            "title": "Bot to Delete",
            "description": "Bot for delete audit test",
            "type": "knowledge",
            "config": json.dumps(config),
            "collection_ids": [collection["id"]],
        }
        
        resp = client.post("/api/v1/bots", json=bot_data)
        assert resp.status_code == HTTPStatus.OK
        bot_id = resp.json()["id"]
        
        # Delete bot
        resp = client.delete(f"/api/v1/bots/{bot_id}")
        assert resp.status_code in [HTTPStatus.OK, HTTPStatus.NO_CONTENT], f"Failed to delete bot: {resp.text}"
        
        # Check audit log
        audit_log = audit_helper.find_audit_log(
            resource_type="bot",
            api_name="DeleteBot",
            resource_id=bot_id,
            http_method="DELETE"
        )
        
        audit_helper.assert_audit_log_content(audit_log, {
            "resource_type": "bot",
            "api_name": "DeleteBot",
            "resource_id": bot_id,
            "http_method": "DELETE",
            "path": f"/api/v1/bots/{bot_id}"
        })


class TestChatAudit:
    """Test audit logs for chat operations"""
    
    def test_create_chat_audit(self, client, audit_helper, bot):
        """Test that creating a chat generates audit log"""
        bot_id = bot["id"]
        
        # Create chat
        resp = client.post(f"/api/v1/bots/{bot_id}/chats")
        assert resp.status_code == HTTPStatus.OK, f"Failed to create chat: {resp.text}"
        chat = resp.json()
        chat_id = chat["id"]
        
        try:
            # Check audit log
            audit_log = audit_helper.find_audit_log(
                resource_type="chat",
                api_name="CreateChat",
                resource_id=chat_id,
                http_method="POST",
                match_response_data={"id": chat_id}
            )
            
            audit_helper.assert_audit_log_content(audit_log, {
                "resource_type": "chat",
                "api_name": "CreateChat",
                "http_method": "POST",
                "path": f"/api/v1/bots/{bot_id}/chats"
            })
            
            # Verify response data contains chat info
            response_data = audit_helper.parse_json_field(audit_log, "response_data")
            assert "id" in response_data
            assert response_data["id"] == chat_id
            
        finally:
            # Cleanup
            client.delete(f"/api/v1/bots/{bot_id}/chats/{chat_id}")
    
    def test_update_chat_audit(self, client, audit_helper, bot):
        """Test that updating a chat generates audit log"""
        bot_id = bot["id"]
        
        # Create chat first
        resp = client.post(f"/api/v1/bots/{bot_id}/chats")
        assert resp.status_code == HTTPStatus.OK
        chat = resp.json()
        chat_id = chat["id"]
        
        try:
            # Update chat
            update_data = {"title": "Updated Chat Title"}
            resp = client.put(f"/api/v1/bots/{bot_id}/chats/{chat_id}", json=update_data)
            assert resp.status_code == HTTPStatus.OK, f"Failed to update chat: {resp.text}"
            
            # Check audit log
            audit_log = audit_helper.find_audit_log(
                resource_type="chat",
                api_name="UpdateChat",
                resource_id=chat_id,
                http_method="PUT"
            )
            
            audit_helper.assert_audit_log_content(audit_log, {
                "resource_type": "chat",
                "api_name": "UpdateChat",
                "resource_id": chat_id,
                "http_method": "PUT",
                "path": f"/api/v1/bots/{bot_id}/chats/{chat_id}"
            })
            
            # Verify request data contains update info
            request_data = audit_helper.parse_json_field(audit_log, "request_data")
            # The request data structure might be nested, check both formats
            if "title" in request_data:
                assert request_data["title"] == update_data["title"]
            elif "chat_in" in request_data and "title" in request_data["chat_in"]:
                assert request_data["chat_in"]["title"] == update_data["title"]
            else:
                assert False, f"title field not found in request_data: {request_data}"
            
        finally:
            # Cleanup
            client.delete(f"/api/v1/bots/{bot_id}/chats/{chat_id}")
    
    def test_delete_chat_audit(self, client, audit_helper, bot):
        """Test that deleting a chat generates audit log"""
        bot_id = bot["id"]
        
        # Create chat first
        resp = client.post(f"/api/v1/bots/{bot_id}/chats")
        assert resp.status_code == HTTPStatus.OK
        chat_id = resp.json()["id"]
        
        # Delete chat
        resp = client.delete(f"/api/v1/bots/{bot_id}/chats/{chat_id}")
        assert resp.status_code in [HTTPStatus.OK, HTTPStatus.NO_CONTENT], f"Failed to delete chat: {resp.text}"
        
        # Check audit log
        audit_log = audit_helper.find_audit_log(
            resource_type="chat",
            api_name="DeleteChat",
            resource_id=chat_id,
            http_method="DELETE"
        )
        
        audit_helper.assert_audit_log_content(audit_log, {
            "resource_type": "chat",
            "api_name": "DeleteChat",
            "resource_id": chat_id,
            "http_method": "DELETE",
            "path": f"/api/v1/bots/{bot_id}/chats/{chat_id}"
        })


class TestLLMProviderAudit:
    """Test audit logs for LLM provider operations"""
    
    def test_create_llm_provider_audit(self, cookie_client, audit_helper):
        """Test that creating an LLM provider generates audit log"""
        # Create LLM provider
        provider_data = {
            "name": "audit-test-provider",
            "label": "Audit Test Provider", 
            "base_url": "https://api.example.com/v1",
            "completion_dialect": "openai",
            "embedding_dialect": "openai",
            "api_key": "test-api-key"
        }
        
        resp = cookie_client.post("/api/v1/llm_providers", json=provider_data)
        assert resp.status_code == HTTPStatus.OK, f"Failed to create LLM provider: {resp.text}"
        provider = resp.json()
        provider_name = provider["name"]
        
        try:
            # Check audit log
            audit_log = audit_helper.find_audit_log(
                resource_type="llm_provider",
                api_name="CreateLLMProvider",
                resource_id=provider_name,
                http_method="POST",
                match_response_data={"name": provider_name}
            )
            
            audit_helper.assert_audit_log_content(audit_log, {
                "resource_type": "llm_provider",
                "api_name": "CreateLLMProvider",
                "http_method": "POST",
                "path": "/api/v1/llm_providers"
            })
            
            # Verify request data (API key should be filtered)
            request_data = audit_helper.parse_json_field(audit_log, "request_data")
            assert "label" in request_data
            assert request_data["label"] == provider_data["label"]
            # API key should be filtered out or redacted
            if "api_key" in request_data:
                assert request_data["api_key"] == "***FILTERED***"
            
        finally:
            # Cleanup
            cookie_client.delete(f"/api/v1/llm_providers/{provider_name}")
    
    def test_update_llm_provider_audit(self, cookie_client, audit_helper):
        """Test that updating an LLM provider generates audit log"""
        # Create provider first
        provider_data = {
            "name": "update-test-provider",
            "label": "Provider for Update Test",
            "base_url": "https://api.example.com/v1",
            "api_key": "initial-key"
        }
        
        resp = cookie_client.post("/api/v1/llm_providers", json=provider_data)
        assert resp.status_code == HTTPStatus.OK
        provider_name = resp.json()["name"]
        
        try:
            # Update provider
            update_data = {
                "label": "Updated Provider Label",
                "base_url": "https://api.updated.com/v1",
                "api_key": "updated-key"
            }
            
            resp = cookie_client.put(f"/api/v1/llm_providers/{provider_name}", json=update_data)
            assert resp.status_code == HTTPStatus.OK, f"Failed to update LLM provider: {resp.text}"
            
            # Check audit log
            audit_log = audit_helper.find_audit_log(
                resource_type="llm_provider",
                api_name="UpdateLLMProvider",
                resource_id=provider_name,
                http_method="PUT"
            )
            
            audit_helper.assert_audit_log_content(audit_log, {
                "resource_type": "llm_provider",
                "api_name": "UpdateLLMProvider",
                "resource_id": provider_name,
                "http_method": "PUT",
                "path": f"/api/v1/llm_providers/{provider_name}"
            })
            
            # Verify request data contains update info (API key should be filtered)
            request_data = audit_helper.parse_json_field(audit_log, "request_data")
            # The request data structure might be nested, check both formats
            if "label" in request_data:
                assert request_data["label"] == update_data["label"]
                if "api_key" in request_data:
                    assert request_data["api_key"] == "***FILTERED***"
            elif "provider_data" in request_data and "label" in request_data["provider_data"]:
                assert request_data["provider_data"]["label"] == update_data["label"]
                if "api_key" in request_data["provider_data"]:
                    assert request_data["provider_data"]["api_key"] == "***FILTERED***"
            else:
                assert False, f"label field not found in request_data: {request_data}"
                
        finally:
            # Cleanup
            cookie_client.delete(f"/api/v1/llm_providers/{provider_name}")
    
    def test_delete_llm_provider_audit(self, cookie_client, audit_helper):
        """Test that deleting an LLM provider generates audit log"""
        # Create provider first
        provider_data = {
            "name": "delete-test-provider",
            "label": "Provider for Delete Test",
            "base_url": "https://api.example.com/v1"
        }
        
        resp = cookie_client.post("/api/v1/llm_providers", json=provider_data)
        assert resp.status_code == HTTPStatus.OK
        provider_name = resp.json()["name"]
        
        # Delete provider
        resp = cookie_client.delete(f"/api/v1/llm_providers/{provider_name}")
        assert resp.status_code in [HTTPStatus.OK, HTTPStatus.NO_CONTENT], f"Failed to delete LLM provider: {resp.text}"
        
        # Check audit log
        audit_log = audit_helper.find_audit_log(
            resource_type="llm_provider",
            api_name="DeleteLLMProvider",
            resource_id=provider_name,
            http_method="DELETE"
        )
        
        audit_helper.assert_audit_log_content(audit_log, {
            "resource_type": "llm_provider",
            "api_name": "DeleteLLMProvider",
            "resource_id": provider_name,
            "http_method": "DELETE",
            "path": f"/api/v1/llm_providers/{provider_name}"
        })


class TestLLMProviderModelAudit:
    """Test audit logs for LLM provider model operations"""
    
    @pytest.fixture
    def test_provider(self, cookie_client):
        """Create a test LLM provider for model tests"""
        provider_data = {
            "name": "model-test-provider",
            "label": "Provider for Model Test",
            "base_url": "https://api.example.com/v1"
        }
        
        resp = cookie_client.post("/api/v1/llm_providers", json=provider_data)
        assert resp.status_code == HTTPStatus.OK
        provider = resp.json()
        
        yield provider
        
        # Cleanup
        cookie_client.delete(f"/api/v1/llm_providers/{provider['name']}")
    
    def test_create_llm_provider_model_audit(self, cookie_client, audit_helper, test_provider):
        """Test that creating an LLM provider model generates audit log"""
        provider_name = test_provider["name"]
        
        # Create provider model
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        model_data = {
            "api": "completion",
            "model": f"test-model-{unique_id}",
            "custom_llm_provider": "openai",
            "max_tokens": 4096,
            "tags": ["test"]
        }
        
        resp = cookie_client.post(f"/api/v1/llm_providers/{provider_name}/models", json=model_data)
        assert resp.status_code == HTTPStatus.OK, f"Failed to create provider model: {resp.text}"
        
        # Check audit log
        audit_log = audit_helper.find_audit_log(
            resource_type="llm_provider_model",
            api_name="CreateProviderModel",
            http_method="POST"
        )
        
        audit_helper.assert_audit_log_content(audit_log, {
            "resource_type": "llm_provider_model",
            "api_name": "CreateProviderModel",
            "http_method": "POST",
            "path": f"/api/v1/llm_providers/{provider_name}/models"
        })
        
        # Verify response data contains model info (since request data extraction has limitations)
        response_data = audit_helper.parse_json_field(audit_log, "response_data")
        assert "model" in response_data
        assert response_data["model"] == model_data["model"]
        assert "custom_llm_provider" in response_data
        assert response_data["custom_llm_provider"] == model_data["custom_llm_provider"]
    
    def test_update_llm_provider_model_audit(self, cookie_client, audit_helper, test_provider):
        """Test that updating an LLM provider model generates audit log"""
        provider_name = test_provider["name"]
        
        # Create model first
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        model_data = {
            "api": "completion",
            "model": f"test-model-{unique_id}",
            "custom_llm_provider": "openai",
            "max_tokens": 4096
        }
        
        resp = cookie_client.post(f"/api/v1/llm_providers/{provider_name}/models", json=model_data)
        assert resp.status_code == HTTPStatus.OK
        
        # Update model
        update_data = {
            "custom_llm_provider": "anthropic",
            "max_tokens": 8192,
            "tags": ["updated", "test"]
        }
        
        api = model_data["api"]
        model = model_data["model"]
        resp = cookie_client.put(f"/api/v1/llm_providers/{provider_name}/models/{api}/{model}", json=update_data)
        assert resp.status_code == HTTPStatus.OK, f"Failed to update provider model: {resp.text}"
        
        # Check audit log
        audit_log = audit_helper.find_audit_log(
            resource_type="llm_provider_model",
            api_name="UpdateProviderModel",
            http_method="PUT"
        )
        
        audit_helper.assert_audit_log_content(audit_log, {
            "resource_type": "llm_provider_model",
            "api_name": "UpdateProviderModel",
            "http_method": "PUT",
            "path": f"/api/v1/llm_providers/{provider_name}/models/{api}/{model}"
        })
        
        # Verify response data contains update info (since request data extraction has limitations)
        response_data = audit_helper.parse_json_field(audit_log, "response_data")
        assert "custom_llm_provider" in response_data
        assert response_data["custom_llm_provider"] == update_data["custom_llm_provider"]
    
    def test_delete_llm_provider_model_audit(self, cookie_client, audit_helper, test_provider):
        """Test that deleting an LLM provider model generates audit log"""
        provider_name = test_provider["name"]
        
        # Create model first
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        model_data = {
            "api": "completion",
            "model": f"test-model-{unique_id}",
            "custom_llm_provider": "openai"
        }
        
        resp = cookie_client.post(f"/api/v1/llm_providers/{provider_name}/models", json=model_data)
        assert resp.status_code == HTTPStatus.OK
        
        # Delete model
        api = model_data["api"]
        model = model_data["model"]
        resp = cookie_client.delete(f"/api/v1/llm_providers/{provider_name}/models/{api}/{model}")
        assert resp.status_code in [HTTPStatus.OK, HTTPStatus.NO_CONTENT], f"Failed to delete provider model: {resp.text}"
        
        # Check audit log
        audit_log = audit_helper.find_audit_log(
            resource_type="llm_provider_model",
            api_name="DeleteProviderModel",
            http_method="DELETE"
        )
        
        audit_helper.assert_audit_log_content(audit_log, {
            "resource_type": "llm_provider_model",
            "api_name": "DeleteProviderModel",
            "http_method": "DELETE",
            "path": f"/api/v1/llm_providers/{provider_name}/models/{api}/{model}"
        })


class TestAuditLogRetrieval:
    """Test audit log retrieval and filtering functionality"""
    
    def test_list_audit_logs(self, cookie_client, audit_helper):
        """Test basic audit log listing functionality"""
        # Get audit logs
        logs = audit_helper.get_audit_logs(limit=10)
        assert isinstance(logs, list), "Audit logs should be a list"
        
        # Verify log structure
        if logs:  # Only check if there are logs
            log = logs[0]
            required_fields = [
                "id", "resource_type", "api_name", "http_method", "path", 
                "status_code", "start_time", "end_time", "created"
            ]
            for field in required_fields:
                assert field in log, f"Field {field} should be present in audit log"
    
    def test_audit_log_filtering(self, cookie_client, audit_helper):
        """Test audit log filtering by resource type and other criteria"""
        # Test filtering by resource type
        collection_logs = audit_helper.get_audit_logs(resource_type="collection")
        if collection_logs:
            for log in collection_logs:
                assert log["resource_type"] == "collection", "All logs should be collection type"
        
        # Test filtering by HTTP method
        post_logs = audit_helper.get_audit_logs(http_method="POST")
        if post_logs:
            for log in post_logs:
                assert log["http_method"] == "POST", "All logs should be POST method"
        
        # Test filtering by status code
        success_logs = audit_helper.get_audit_logs(status_code=200)
        if success_logs:
            for log in success_logs:
                assert log["status_code"] == 200, "All logs should have status code 200"
    
    def test_audit_log_detail(self, cookie_client, audit_helper):
        """Test retrieving individual audit log details"""
        # Get a log first
        logs = audit_helper.get_audit_logs(limit=1)
        if not logs:
            pytest.skip("No audit logs available for testing")
        
        log_id = logs[0]["id"]
        
        # Get detailed log
        resp = cookie_client.get(f"/api/v1/audit-logs/{log_id}")
        assert resp.status_code == HTTPStatus.OK, f"Failed to get audit log detail: {resp.text}"
        
        detailed_log = resp.json()
        assert detailed_log["id"] == log_id, "Log ID should match"
        
        # Verify all expected fields are present
        required_fields = [
            "id", "resource_type", "api_name", "http_method", "path",
            "status_code", "start_time", "end_time", "created"
        ]
        for field in required_fields:
            assert field in detailed_log, f"Field {field} should be present in detailed audit log"


class TestAuditSensitiveDataFiltering:
    """Test that sensitive data is properly filtered in audit logs"""
    
    def test_api_key_filtering_in_audit(self, cookie_client, audit_helper):
        """Test that API keys are filtered in audit logs"""
        # Create LLM provider with API key
        provider_data = {
            "name": "sensitive-test-provider",
            "label": "Sensitive Data Test Provider",
            "base_url": "https://api.example.com/v1",
            "api_key": "sk-very-secret-api-key-12345"
        }
        
        resp = cookie_client.post("/api/v1/llm_providers", json=provider_data)
        assert resp.status_code == HTTPStatus.OK
        provider_name = resp.json()["name"]
        
        try:
            # Find the audit log
            audit_log = audit_helper.find_audit_log(
                resource_type="llm_provider",
                api_name="CreateLLMProvider",
                resource_id=provider_name,
                http_method="POST"
            )
            
            # Verify API key is filtered
            request_data = audit_helper.parse_json_field(audit_log, "request_data")
            if "api_key" in request_data:
                # API key should be filtered out or replaced with a placeholder
                assert request_data["api_key"] != provider_data["api_key"], "API key should be filtered"
                assert request_data["api_key"] == "***FILTERED***", "API key should be replaced with placeholder"
            
        finally:
            # Cleanup
            cookie_client.delete(f"/api/v1/llm_providers/{provider_name}")


class TestAuditLogIntegrity:
    """Test audit log data integrity and consistency"""
    
    def test_audit_log_timestamps(self, cookie_client, audit_helper):
        """Test that audit log timestamps are consistent and valid"""
        # Create a simple resource to generate audit log
        provider_data = {
            "name": "timestamp-test-provider",
            "label": "Timestamp Test Provider",
            "base_url": "https://api.example.com/v1"
        }
        
        before_request = int(time.time() * 1000)  # milliseconds
        resp = cookie_client.post("/api/v1/llm_providers", json=provider_data)
        after_request = int(time.time() * 1000)  # milliseconds
        
        assert resp.status_code == HTTPStatus.OK
        provider_name = resp.json()["name"]
        
        try:
            # Find the audit log
            audit_log = audit_helper.find_audit_log(
                resource_type="llm_provider",
                api_name="CreateLLMProvider",
                resource_id=provider_name,
                http_method="POST",
                match_response_data={"name": provider_name}
            )
            
            # Verify timestamps
            start_time = audit_log.get("start_time")
            end_time = audit_log.get("end_time")
            duration_ms = audit_log.get("duration_ms")
            
            assert start_time is not None, "start_time should not be None"
            assert end_time is not None, "end_time should not be None"
            assert duration_ms is not None, "duration_ms should not be None"
            
            # Verify timestamp ranges
            assert before_request <= start_time <= after_request, "start_time should be within request timeframe"
            assert before_request <= end_time <= after_request + 5000, "end_time should be reasonable"  # Allow 5s buffer
            
            # Verify duration calculation
            expected_duration = end_time - start_time
            assert duration_ms == expected_duration, f"duration_ms should match calculated duration: {duration_ms} != {expected_duration}"
            
            # Verify start_time <= end_time
            assert start_time <= end_time, "start_time should be <= end_time"
            
        finally:
            # Cleanup
            cookie_client.delete(f"/api/v1/llm_providers/{provider_name}")
    
    def test_audit_log_user_info(self, cookie_client, audit_helper):
        """Test that audit logs contain correct user information"""
        # Create a simple resource to generate audit log
        provider_data = {
            "name": "user-info-test-provider",
            "label": "User Info Test Provider",
            "base_url": "https://api.example.com/v1"
        }
        
        resp = cookie_client.post("/api/v1/llm_providers", json=provider_data)
        assert resp.status_code == HTTPStatus.OK
        provider_name = resp.json()["name"]
        
        try:
            # Find the audit log
            audit_log = audit_helper.find_audit_log(
                resource_type="llm_provider",
                api_name="CreateLLMProvider",
                resource_id=provider_name,
                http_method="POST",
                match_response_data={"name": provider_name}
            )
            
            # Verify user information is present
            assert audit_log.get("user_id") is not None, "user_id should be present"
            assert audit_log.get("username") is not None, "username should be present"
            
            # Verify user_id is a valid UUID-like string
            user_id = audit_log.get("user_id")
            assert len(user_id) > 0, "user_id should not be empty"
            
            # Verify username is a valid string
            username = audit_log.get("username")
            assert len(username) > 0, "username should not be empty"
            assert isinstance(username, str), "username should be a string"
            
        finally:
            # Cleanup
            cookie_client.delete(f"/api/v1/llm_providers/{provider_name}")


# Additional helper functions for testing audit logs in specific scenarios

def verify_audit_log_for_resource_operation(audit_helper, resource_type: str, operation: str, 
                                           resource_id: str = None, expected_path: str = None):
    """Helper function to verify audit log for a specific resource operation"""
    http_method_map = {
        "create": "POST",
        "update": "PUT", 
        "delete": "DELETE"
    }
    
    api_name_map = {
        "collection": {
            "create": "CreateCollection",
            "update": "UpdateCollection", 
            "delete": "DeleteCollection"
        },
        "document": {
            "create": "CreateDocuments",
            "update": "UpdateDocument",
            "delete": "DeleteDocument"
        },
        "bot": {
            "create": "CreateBot",
            "update": "UpdateBot",
            "delete": "DeleteBot"
        },
        "chat": {
            "create": "CreateChat",
            "update": "UpdateChat",
            "delete": "DeleteChat"
        },
        "llm_provider": {
            "create": "CreateLLMProvider",
            "update": "UpdateLLMProvider", 
            "delete": "DeleteLLMProvider"
        },
        "llm_provider_model": {
            "create": "CreateProviderModel",
            "update": "UpdateProviderModel",
            "delete": "DeleteProviderModel"
        }
    }
    
    http_method = http_method_map.get(operation)
    api_name = api_name_map.get(resource_type, {}).get(operation)
    
    if not http_method or not api_name:
        raise ValueError(f"Invalid resource_type '{resource_type}' or operation '{operation}'")
    
    audit_log = audit_helper.find_audit_log(
        resource_type=resource_type,
        api_name=api_name,
        resource_id=resource_id,
        http_method=http_method
    )
    
    expected_fields = {
        "resource_type": resource_type,
        "api_name": api_name,
        "http_method": http_method
    }
    
    if resource_id:
        expected_fields["resource_id"] = resource_id
    
    if expected_path:
        expected_fields["path"] = expected_path
    
    audit_helper.assert_audit_log_content(audit_log, expected_fields)
    
    return audit_log


"""
Test Usage Examples:

1. Run all audit tests:
   pytest tests/e2e_test/test_audit.py -v

2. Run specific test class:
   pytest tests/e2e_test/test_audit.py::TestCollectionAudit -v

3. Run specific test method:
   pytest tests/e2e_test/test_audit.py::TestCollectionAudit::test_create_collection_audit -v

4. Run tests with specific markers (if added):
   pytest tests/e2e_test/test_audit.py -m "audit" -v

5. Run audit tests for specific resource types:
   pytest tests/e2e_test/test_audit.py::TestCollectionAudit -v  # Collection tests
   pytest tests/e2e_test/test_audit.py::TestBotAudit -v        # Bot tests
   pytest tests/e2e_test/test_audit.py::TestChatAudit -v       # Chat tests

The test cases cover:
- Creation/Update/Deletion operations for all specified resource types
- Audit log content validation (resource_type, api_name, timestamps, etc.)
- Sensitive data filtering (API keys, passwords, etc.)
- Audit log retrieval and filtering functionality
- Data integrity and consistency checks
- User information tracking in audit logs

Each test follows the pattern:
1. Perform operation that should generate audit log
2. Search for the audit log using the helper
3. Validate audit log content and structure
4. Clean up resources created during test

The AuditLogTestHelper class provides convenient methods for:
- Retrieving audit logs with filters
- Finding specific audit logs by criteria
- Asserting audit log content matches expectations
- Handling retry logic for eventual consistency
"""
