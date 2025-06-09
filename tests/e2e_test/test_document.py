import io
import time
from http import HTTPStatus

import pytest

from tests.e2e_test.config import MAX_DOCUMENT_SIZE_MB


def make_markdown_file():
    # Simulate a markdown file
    return "test1.md", io.BytesIO(b"# Title\nThis is a test markdown file."), "text/markdown"


def make_large_markdown_file(size_mb=100):
    content = "# Title\n" + "This is a test markdown file." * (size_mb * 1024 * 1024)
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
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 3


@pytest.mark.parametrize("size_mb", [105, 10])
def test_upload_large_document(client, collection, size_mb):
    """Test uploading a large document (e.g., >10MB markdown)"""
    files = [
        ("files", make_large_markdown_file(size_mb)),
    ]
    response = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    if size_mb >= MAX_DOCUMENT_SIZE_MB:
        assert response.status_code == HTTPStatus.BAD_REQUEST
    else:
        assert response.status_code == HTTPStatus.OK


def test_upload_unsupported_file_type(benchmark, client, collection):
    """Test uploading an unsupported file type (e.g., .exe)"""
    files = [
        ("files", make_exe_file()),
    ]
    response = benchmark(client.post, f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert response.status_code == HTTPStatus.BAD_REQUEST


def test_upload_too_many_documents(client, collection):
    """Test uploading more than the allowed number of documents (e.g., 100)"""
    filename, file_obj, mimetype = make_markdown_file()
    files = [("files", (f"test_{i}.md", io.BytesIO(file_obj.getvalue()), mimetype)) for i in range(100)]
    response = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert response.status_code == HTTPStatus.BAD_REQUEST


def test_upload_duplicate_then_delete_and_reupload(client, collection):
    """Test uploading a duplicate document, delete it, then re-upload successfully"""
    # Step 1: Upload a markdown file (should succeed)
    filename, file_obj, mimetype = make_markdown_file()
    files = [("files", (filename, file_obj, mimetype))]
    response = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert response.status_code == HTTPStatus.OK
    data = response.json()
    assert "items" in data
    assert len(data["items"]) == 1
    doc_id = data["items"][0]["id"]

    # Step 2: Upload the same file name again (should fail due to duplicate)
    file_obj2 = io.BytesIO(file_obj.getvalue())
    files2 = [("files", (filename, file_obj2, mimetype))]
    response2 = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files2)
    assert response2.status_code == HTTPStatus.BAD_REQUEST

    # Step 3: Delete the original document
    del_response = client.delete(f"/api/v1/collections/{collection['id']}/documents/{doc_id}")
    assert del_response.status_code == HTTPStatus.OK
    # check if the document is deleted
    max_wait = 10
    interval = 2
    for _ in range(max_wait // interval):
        list_response = client.get(f"/api/v1/collections/{collection['id']}/documents")
        assert list_response.status_code == HTTPStatus.OK
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
    assert response3.status_code == HTTPStatus.OK
    data3 = response3.json()
    assert "items" in data3
    assert len(data3["items"]) == 1
    assert data3["items"][0]["name"] == filename
