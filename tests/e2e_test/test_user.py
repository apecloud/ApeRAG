from http import HTTPStatus

import httpx

def test_change_password(cookie_client, login_user):
    """Test change password for user"""
    data = {
        "username": login_user["username"],
        "old_password": login_user["password"],
        "new_password": login_user["password"] + "_new",
    }
    resp = cookie_client.post("/api/v1/change-password", json=data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    user = resp.json()
    assert user["username"] == login_user["username"]
    # Try login with new password
    with httpx.Client(base_url=cookie_client.base_url) as c:
        resp2 = c.post("/api/v1/login", json={"username": login_user["username"], "password": data["new_password"]})
        assert resp2.status_code == HTTPStatus.OK, resp2.text


def test_get_user_list(benchmark, cookie_client):
    """Test get user list (should be empty or only self if not admin)"""
    resp = benchmark(cookie_client.get, "/api/v1/users")
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_delete_user(cookie_client, login_user):
    """Test delete user (should fail for non-admin, succeed for admin)"""
    # Try to delete self (should fail)
    user_id = login_user["user"]["id"]
    resp = cookie_client.delete(f"/api/v1/users/{user_id}")
    assert resp.status_code != HTTPStatus.OK, (
        f"Should not be able to delete self, but got {resp.status_code}: {resp.text}"
    )
