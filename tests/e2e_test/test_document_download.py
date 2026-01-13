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

"""
End-to-end tests for document download functionality
"""

import time
from http import HTTPStatus


def test_download_document(client, collection):
    """Test downloading a document file"""
    # Upload a test document
    test_content = b"This is a test document for download testing. Hello ApeRAG!"
    files = {"files": ("test_download.txt", test_content, "text/plain")}
    upload_resp = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert upload_resp.status_code == HTTPStatus.OK, upload_resp.text
    resp_data = upload_resp.json()
    assert len(resp_data["items"]) == 1
    doc_id = resp_data["items"][0]["id"]

    # Wait for document to be processed
    max_wait = 30
    interval = 2
    for _ in range(max_wait // interval):
        get_resp = client.get(f"/api/v1/collections/{collection['id']}/documents/{doc_id}")
        assert get_resp.status_code == HTTPStatus.OK, get_resp.text
        data = get_resp.json()
        status = data.get("status")
        if status in ["COMPLETE", "FAILED", "RUNNING"]:
            break
        time.sleep(interval)

    # Download the document
    download_resp = client.get(f"/api/v1/collections/{collection['id']}/documents/{doc_id}/download")
    assert download_resp.status_code == HTTPStatus.OK, download_resp.text

    # Verify response headers
    assert "content-type" in download_resp.headers
    assert "content-disposition" in download_resp.headers
    assert "attachment" in download_resp.headers["content-disposition"]
    assert "test_download.txt" in download_resp.headers["content-disposition"]

    # Verify content
    downloaded_content = download_resp.content
    assert downloaded_content == test_content, "Downloaded content should match uploaded content"

    # Cleanup: Delete document
    delete_resp = client.delete(f"/api/v1/collections/{collection['id']}/documents/{doc_id}")
    assert delete_resp.status_code == HTTPStatus.OK, delete_resp.text


def test_download_nonexistent_document(client, collection):
    """Test downloading a non-existent document"""
    fake_doc_id = "doc_nonexistent12345"
    download_resp = client.get(f"/api/v1/collections/{collection['id']}/documents/{fake_doc_id}/download")
    assert download_resp.status_code == HTTPStatus.NOT_FOUND, download_resp.text


def test_download_deleted_document(client, collection):
    """Test downloading a deleted document"""
    # Upload a test document
    test_content = b"This document will be deleted before download."
    files = {"files": ("test_deleted.txt", test_content, "text/plain")}
    upload_resp = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert upload_resp.status_code == HTTPStatus.OK, upload_resp.text
    resp_data = upload_resp.json()
    assert len(resp_data["items"]) == 1
    doc_id = resp_data["items"][0]["id"]

    # Wait for document to be processed
    max_wait = 30
    interval = 2
    for _ in range(max_wait // interval):
        get_resp = client.get(f"/api/v1/collections/{collection['id']}/documents/{doc_id}")
        assert get_resp.status_code == HTTPStatus.OK, get_resp.text
        data = get_resp.json()
        status = data.get("status")
        if status in ["COMPLETE", "FAILED", "RUNNING"]:
            break
        time.sleep(interval)

    # Delete the document
    delete_resp = client.delete(f"/api/v1/collections/{collection['id']}/documents/{doc_id}")
    assert delete_resp.status_code == HTTPStatus.OK, delete_resp.text

    # Try to download the deleted document (should fail)
    download_resp = client.get(f"/api/v1/collections/{collection['id']}/documents/{doc_id}/download")
    assert download_resp.status_code == HTTPStatus.NOT_FOUND, download_resp.text


def test_download_pdf_document(client, collection):
    """Test downloading a PDF document with correct content type"""
    # Create a minimal PDF content (just for testing headers, not a real PDF)
    # In real scenario, you would upload an actual PDF file
    test_pdf_content = b"%PDF-1.4\n%Test PDF content\n%%EOF"
    files = {"files": ("test_document.pdf", test_pdf_content, "application/pdf")}
    upload_resp = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert upload_resp.status_code == HTTPStatus.OK, upload_resp.text
    resp_data = upload_resp.json()
    assert len(resp_data["items"]) == 1
    doc_id = resp_data["items"][0]["id"]

    # Wait for document to be processed
    max_wait = 30
    interval = 2
    for _ in range(max_wait // interval):
        get_resp = client.get(f"/api/v1/collections/{collection['id']}/documents/{doc_id}")
        assert get_resp.status_code == HTTPStatus.OK, get_resp.text
        data = get_resp.json()
        status = data.get("status")
        if status in ["COMPLETE", "FAILED", "RUNNING"]:
            break
        time.sleep(interval)

    # Download the document
    download_resp = client.get(f"/api/v1/collections/{collection['id']}/documents/{doc_id}/download")
    assert download_resp.status_code == HTTPStatus.OK, download_resp.text

    # Verify response headers - content type should be PDF
    assert "content-type" in download_resp.headers
    assert "pdf" in download_resp.headers["content-type"].lower()
    assert "content-disposition" in download_resp.headers
    assert "test_document.pdf" in download_resp.headers["content-disposition"]

    # Cleanup: Delete document
    delete_resp = client.delete(f"/api/v1/collections/{collection['id']}/documents/{doc_id}")
    assert delete_resp.status_code == HTTPStatus.OK, delete_resp.text


def test_download_unauthorized_access(client, collection):
    """Test downloading a document from another user's collection (should fail)"""
    # This test assumes there's a way to create documents under different users
    # For now, we just test that the endpoint requires authentication
    # by attempting to access with wrong collection_id
    fake_collection_id = "col_unauthorized123"
    fake_doc_id = "doc_unauthorized123"

    download_resp = client.get(f"/api/v1/collections/{fake_collection_id}/documents/{fake_doc_id}/download")
    # Should return 404 (not found) or 403 (forbidden) depending on implementation
    assert download_resp.status_code in [HTTPStatus.NOT_FOUND, HTTPStatus.FORBIDDEN], download_resp.text
