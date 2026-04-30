"""User-facing authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.auth import get_current_user, verify_credentials
from app.deps import STATIC_DIR

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


@router.get("/login")
async def login_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "login.html")


@router.post("/api/login")
async def api_login(request: Request, body: LoginRequest):
    account, is_admin = verify_credentials(body.email, body.password)
    if account is None:
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)
    request.session["email"] = account.email
    request.session["display_name"] = account.display_name
    request.session["logo_url"] = account.logo_url
    if is_admin:
        request.session["is_admin"] = True
    return {"ok": True, "display_name": account.display_name, "logo_url": account.logo_url}


@router.post("/api/logout")
async def api_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/api/me")
async def api_me(request: Request):
    user = get_current_user(request)
    if user is None:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    return user
