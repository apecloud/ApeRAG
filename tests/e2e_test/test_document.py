import io
import time
from http import HTTPStatus

import pytest


def make_markdown_file():
    # Simulate a markdown file
    return "test1.md", io.BytesIO(b"# Title\nThis is a test markdown file."), "text/markdown"


def make_large_markdown_file(size_mb=100):
    content = "# Title\nThis is a test markdown file.\n" + "a" * (size_mb * 1024 * 1024)
    return "large_file.md", io.BytesIO(content.encode("utf-8")), "text/markdown"


def make_exe_file():
    # Simulate an exe file
    return "malware.exe", io.BytesIO(b"MZP\x00\x02..."), "application/octet-stream"


def test_upload_multiple_documents(client, collection):
    """Test uploading multiple documents at once"""
    files = [
        ("files", make_markdown_file()),
        ("files", make_markdown_file()),
        ("files", make_markdown_file()),
    ]
    response = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert response.status_code == HTTPStatus.OK, response.text
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 3


def test_upload_document_and_wait_to_be_active(client, collection):
    files = [
        ("files", make_markdown_file()),
    ]
    response = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert response.status_code == HTTPStatus.OK, response.text
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    doc_id = data["items"][0]["id"]
    max_wait = 30
    interval = 2
    for _ in range(max_wait // interval):
        list_response = client.get(f"/api/v1/collections/{collection['id']}/documents")
        assert list_response.status_code == HTTPStatus.OK, list_response.text
        list_data = list_response.json()
        for doc in list_data.get("items", []):
            if doc["id"] == doc_id and doc["status"].lower() == "complete":
                return
        time.sleep(interval)
    else:
        pytest.fail("Document is not active")


@pytest.mark.parametrize("size_mb", [105])
def test_upload_large_document(client, collection, size_mb):
    """Test uploading a large document (e.g., >10MB markdown)"""
    files = [
        ("files", make_large_markdown_file(size_mb)),
    ]
    response = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert response.status_code == HTTPStatus.BAD_REQUEST, response.text


def test_upload_unsupported_file_type(benchmark, client, collection):
    """Test uploading an unsupported file type (e.g., .exe)"""
    files = [
        ("files", make_exe_file()),
    ]
    response = benchmark(client.post, f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert response.status_code == HTTPStatus.BAD_REQUEST, response.text


def test_upload_too_many_documents(client, collection):
    """Test uploading more than the allowed number of documents (e.g., 100)"""
    filename, file_obj, mimetype = make_markdown_file()
    files = [("files", (f"test_{i}.md", io.BytesIO(file_obj.getvalue()), mimetype)) for i in range(100)]
    response = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert response.status_code == HTTPStatus.BAD_REQUEST, response.text


@pytest.mark.skipif(True, reason="FIXME: The unique constraint in table document is not work as expected")
def test_upload_duplicate_then_delete_and_reupload(client, collection):
    """Test uploading a duplicate document, delete it, then re-upload successfully"""
    # Step 1: Upload a markdown file (should succeed)
    filename, file_obj, mimetype = make_markdown_file()
    files = [("files", (filename, file_obj, mimetype))]
    response = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert response.status_code == HTTPStatus.OK, response.text
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    doc_id = data["items"][0]["id"]

    # Step 2: Upload the same file name again (should fail due to duplicate)
    file_obj2 = io.BytesIO(file_obj.getvalue())
    files2 = [("files", (filename, file_obj2, mimetype))]
    response2 = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files2)
    assert response2.status_code == HTTPStatus.BAD_REQUEST, response2.text

    # Step 3: Delete the original document
    del_response = client.delete(f"/api/v1/collections/{collection['id']}/documents/{doc_id}")
    assert del_response.status_code == HTTPStatus.OK, del_response.text
    # check if the document is deleted
    max_wait = 10
    interval = 2
    for _ in range(max_wait // interval):
        list_response = client.get(f"/api/v1/collections/{collection['id']}/documents")
        assert list_response.status_code == HTTPStatus.OK, list_response.text
        list_data = list_response.json()
        if all(doc["id"] != doc_id for doc in list_data.get("items", [])):
            break
        time.sleep(interval)
    else:
        pytest.fail("Document is not deleted")

    # Step 4: Re-upload the same file name (should succeed)
    file_obj3 = io.BytesIO(file_obj.getvalue())
    files3 = [("files", (filename, file_obj3, mimetype))]
    response3 = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files3)
    assert response3.status_code == HTTPStatus.OK, response3.text
    data3 = response3.json()
    assert "items" in data3
    assert len(data3["items"]) == 1
    assert data3["items"][0]["name"] == filename


# Document index rebuild tests
def test_rebuild_single_index_type(client, document, collection):
    """Test rebuilding a single index type (vector)"""
    doc_id = document["id"]
    collection_id = collection["id"]
    
    # Test rebuilding vector index
    rebuild_request = {"index_types": ["vector"]}
    response = client.post(
        f"/api/v1/collections/{collection_id}/documents/{doc_id}/rebuild_indexes",
        json=rebuild_request
    )
    assert response.status_code == HTTPStatus.OK, response.text
    data = response.json()
    assert data["code"] == "200"
    assert "vector" in data["message"]


def test_rebuild_all_index_types(client, document, collection):
    """Test rebuilding all supported index types"""
    doc_id = document["id"]
    collection_id = collection["id"]
    
    # Test rebuilding all index types
    rebuild_request = {"index_types": ["vector", "fulltext", "graph"]}
    response = client.post(
        f"/api/v1/collections/{collection_id}/documents/{doc_id}/rebuild_indexes",
        json=rebuild_request
    )
    assert response.status_code == HTTPStatus.OK, response.text
    data = response.json()
    assert data["code"] == "200"
    assert "vector" in data["message"]
    assert "fulltext" in data["message"]
    assert "graph" in data["message"]


def test_rebuild_index_invalid_index_type(client, document, collection):
    """Test rebuilding with invalid index type"""
    doc_id = document["id"]
    collection_id = collection["id"]
    
    # Test with invalid index type
    rebuild_request = {"index_types": ["invalid_type"]}
    response = client.post(
        f"/api/v1/collections/{collection_id}/documents/{doc_id}/rebuild_indexes",
        json=rebuild_request
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, response.text


def test_rebuild_index_empty_index_types(client, document, collection):
    """Test rebuilding with empty index types array"""
    doc_id = document["id"]
    collection_id = collection["id"]
    
    # Test with empty index types
    rebuild_request = {"index_types": []}
    response = client.post(
        f"/api/v1/collections/{collection_id}/documents/{doc_id}/rebuild_indexes",
        json=rebuild_request
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, response.text


@pytest.mark.parametrize("index_type", ["vector", "fulltext", "graph"])
def test_rebuild_individual_index_types(client, document, collection, index_type):
    """Test rebuilding each individual index type"""
    doc_id = document["id"]
    collection_id = collection["id"]
    
    rebuild_request = {"index_types": [index_type]}
    response = client.post(
        f"/api/v1/collections/{collection_id}/documents/{doc_id}/rebuild_indexes",
        json=rebuild_request
    )
    assert response.status_code == HTTPStatus.OK, response.text
    data = response.json()
    assert data["code"] == "200"
    assert index_type in data["message"]


def test_rebuild_index_duplicate_types(client, document, collection):
    """Test rebuilding with duplicate index types in request"""
    doc_id = document["id"]
    collection_id = collection["id"]
    
    # Test with duplicate index types
    rebuild_request = {"index_types": ["vector", "vector", "fulltext"]}
    response = client.post(
        f"/api/v1/collections/{collection_id}/documents/{doc_id}/rebuild_indexes",
        json=rebuild_request
    )
    # Should fail
    assert response.status_code == HTTPStatus.BAD_REQUEST, response.text


def test_rebuild_index_case_sensitivity(client, document, collection):
    """Test rebuilding with different case index types"""
    doc_id = document["id"]
    collection_id = collection["id"]
    
    # Test with uppercase index type (should fail)
    rebuild_request = {"index_types": ["VECTOR"]}
    response = client.post(
        f"/api/v1/collections/{collection_id}/documents/{doc_id}/rebuild_indexes",
        json=rebuild_request
    )
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, response.text
