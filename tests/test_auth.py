"""Tests for demo authentication system."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_models():
    """Mock ASR and MT models to avoid loading them."""
    with (
        patch("app.main.get_asr_backend") as mock_asr,
        patch("app.main.get_translation_backend") as mock_mt,
        patch("app.main.get_summarization_backend") as mock_summ,
    ):
        mock_asr.return_value = MagicMock()
        mock_mt.return_value = MagicMock()
        mock_summ.return_value = None
        yield


@pytest.fixture
def client(mock_models):  # noqa: ARG001
    from app.main import app

    with TestClient(app) as client:
        yield client


def _login(
    client: TestClient,
    email: str = "anna.mueller@synia.de",
    password: str = "Synia#2024!",
):
    """Helper to log in and return the response."""
    return client.post("/api/login", json={"email": email, "password": password})


class TestLogin:
    def test_login_valid_credentials(self, client):
        resp = _login(client)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["display_name"] == "SYNIA Solutions"
        assert data["logo_url"] == "/static/logos/synia.png"

    def test_login_invalid_password(self, client):
        resp = _login(client, password="wrong")
        assert resp.status_code == 401
        assert resp.json()["error"] == "Invalid credentials"

    def test_login_invalid_email(self, client):
        resp = _login(client, email="nobody@example.com")
        assert resp.status_code == 401

    def test_login_sets_session_cookie(self, client):
        resp = _login(client)
        assert resp.status_code == 200
        assert "linguagap_session" in resp.cookies


class TestLogout:
    def test_logout_clears_session(self, client):
        _login(client)
        resp = client.post("/api/logout")
        assert resp.status_code == 200
        me = client.get("/api/me")
        assert me.status_code == 401


class TestApiMe:
    def test_me_without_auth(self, client):
        resp = client.get("/api/me")
        assert resp.status_code == 401

    def test_me_with_auth(self, client):
        _login(client)
        resp = client.get("/api/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "anna.mueller@synia.de"
        assert data["display_name"] == "SYNIA Solutions"


class TestProtectedRoutes:
    def test_root_redirects_without_auth(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/login"

    def test_root_serves_page_with_auth(self, client):
        _login(client)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_metrics_requires_auth(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 401

    def test_metrics_works_with_auth(self, client):
        _login(client)
        resp = client.get("/metrics")
        assert resp.status_code == 200


class TestPublicRoutes:
    def test_health_unprotected(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_login_page_unprotected(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200

    def test_viewer_unprotected(self, client):
        resp = client.get("/viewer/sometoken")
        assert resp.status_code == 200
