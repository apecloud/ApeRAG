from http import HTTPStatus

import httpx

from tests.e2e_test.config import API_BASE_URL


def test_register_user(register_user):
    """Test user registration"""
    user = register_user["user"]
    assert user["username"] == register_user["username"]
    assert user["email"] == register_user["email"]
    assert user["role"] in ("admin", "rw", "ro")
    assert user["is_active"] is True
    assert user["date_joined"]


def test_login_user(login_user):
    """Test user login with cookie"""
    user = login_user["user"]
    assert user["username"] == login_user["username"]
    assert user["email"]
    assert user["role"]
    assert user["is_active"] is True
    assert user["date_joined"]


def test_get_user_detail(benchmark, cookie_client):
    """Test get user detail with cookie authentication"""
    resp = benchmark(cookie_client.get, "/api/v1/user")
    assert resp.status_code == HTTPStatus.OK
    user = resp.json()
    assert user["username"]
    assert user["email"]
    assert user["role"]
    assert user["is_active"] is True
    assert user["date_joined"]


def test_change_password(cookie_client, login_user):
    """Test change password for user"""
    data = {
        "username": login_user["username"],
        "old_password": login_user["password"],
        "new_password": login_user["password"] + "_new",
    }
    resp = cookie_client.post("/api/v1/change-password", json=data)
    assert resp.status_code == HTTPStatus.OK
    user = resp.json()
    assert user["username"] == login_user["username"]
    # Try login with new password
    with httpx.Client(base_url=cookie_client.base_url) as c:
        resp2 = c.post("/api/v1/login", json={"username": login_user["username"], "password": data["new_password"]})
        assert resp2.status_code == HTTPStatus.OK


def test_logout_user(login_user):
    """Test user logout with cookie authentication (use isolated client)"""
    with httpx.Client(base_url=API_BASE_URL) as c:
        c.cookies.update(login_user["cookies"])
        resp = c.post("/api/v1/logout")
        assert resp.status_code == HTTPStatus.OK
        # After logout, get user should fail (unauthorized or forbidden)
        resp2 = c.get("/api/v1/user")
        assert resp2.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN)


def test_get_user_list(benchmark, cookie_client):
    """Test get user list (should be empty or only self if not admin)"""
    resp = benchmark(cookie_client.get, "/api/v1/users")
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_delete_user(cookie_client, login_user):
    """Test delete user (should fail for non-admin, succeed for admin)"""
    # Try to delete self (should fail)
    user_id = login_user["user"]["id"]
    resp = cookie_client.delete(f"/api/v1/users/{user_id}")
    assert resp.status_code != HTTPStatus.OK
