"""Administrative routes — login, account CRUD, logo upload."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from app.auth import LOGOS_DIR, DemoAccount, get_accounts, save_accounts, verify_admin
from app.deps import STATIC_DIR, require_admin
from app.routes.auth import LoginRequest

router = APIRouter()

ALLOWED_LOGO_TYPES = {"image/png", "image/jpeg", "image/svg+xml", "image/webp"}
MAX_LOGO_SIZE = 512 * 1024  # ~500 KB


class AccountRequest(BaseModel):
    email: str
    password: str
    display_name: str
    logo_url: str = "/static/logos/synia.png"


# ---------------------------------------------------------------------------
# Admin login flow + landing page
# ---------------------------------------------------------------------------


@router.get("/admin/login")
async def admin_login_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin-login.html")


@router.post("/api/admin/login")
async def api_admin_login(request: Request, body: LoginRequest):
    if not verify_admin(body.email, body.password):
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)
    request.session["is_admin"] = True
    return {"ok": True}


@router.post("/api/admin/logout", dependencies=[Depends(require_admin)])
async def api_admin_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/admin")
async def admin_page(request: Request):
    if not request.session.get("is_admin"):
        return RedirectResponse("/admin/login", status_code=302)
    return FileResponse(STATIC_DIR / "admin.html")


# ---------------------------------------------------------------------------
# Demo account CRUD
# ---------------------------------------------------------------------------


@router.get("/api/admin/accounts", dependencies=[Depends(require_admin)])
async def list_accounts():
    return [asdict(a) for a in get_accounts()]


@router.post("/api/admin/accounts", dependencies=[Depends(require_admin)])
async def create_account(body: AccountRequest):
    accounts = get_accounts()
    if any(a.email == body.email for a in accounts):
        raise HTTPException(status_code=409, detail="Account with this email already exists")
    account = DemoAccount(
        email=body.email,
        password=body.password,
        display_name=body.display_name,
        logo_url=body.logo_url,
    )
    accounts.append(account)
    save_accounts(accounts)
    return asdict(account)


@router.put("/api/admin/accounts/{email}", dependencies=[Depends(require_admin)])
async def update_account(email: str, body: AccountRequest):
    accounts = get_accounts()
    for i, a in enumerate(accounts):
        if a.email == email:
            accounts[i] = DemoAccount(
                email=body.email,
                password=body.password,
                display_name=body.display_name,
                logo_url=body.logo_url,
            )
            save_accounts(accounts)
            return asdict(accounts[i])
    raise HTTPException(status_code=404, detail="Account not found")


@router.delete("/api/admin/accounts/{email}", dependencies=[Depends(require_admin)])
async def delete_account(email: str):
    accounts = get_accounts()
    new_accounts = [a for a in accounts if a.email != email]
    if len(new_accounts) == len(accounts):
        raise HTTPException(status_code=404, detail="Account not found")
    save_accounts(new_accounts)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Logo upload + serving
# ---------------------------------------------------------------------------


@router.post("/api/admin/upload-logo", dependencies=[Depends(require_admin)])
async def upload_logo(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_LOGO_TYPES:
        raise HTTPException(status_code=400, detail="Only PNG, JPEG, SVG, and WebP images allowed")
    content = await file.read()
    if len(content) > MAX_LOGO_SIZE:
        raise HTTPException(status_code=400, detail="Logo must be under 500KB")
    ext = Path(file.filename or "logo.png").suffix or ".png"
    filename = f"{uuid.uuid4().hex[:12]}{ext}"
    logo_path = LOGOS_DIR / filename
    logo_path.write_bytes(content)
    return {"logo_url": f"/logos/{filename}"}


@router.get("/logos/{filename}")
async def serve_logo(filename: str):
    path = LOGOS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(path)
