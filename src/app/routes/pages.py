"""Static-page and infrastructure endpoints (root, viewer, health, metrics)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, RedirectResponse

from app.deps import STATIC_DIR, require_auth
from app.streaming import get_metrics

router = APIRouter()


@router.get("/")
async def root(request: Request):
    if not request.session.get("email"):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(STATIC_DIR / "index.html")


@router.get("/translate", dependencies=[Depends(require_auth)])
async def translate_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "translate.html")


@router.get("/viewer/{token}")
async def viewer_page(token: str):  # noqa: ARG001 - token is consumed by the WS endpoint
    """Serve the mobile viewer HTML.

    The token in the path is validated client-side via the WebSocket
    connection at /ws/viewer/{token}; this handler only needs to ship the
    static page.
    """
    viewer_html = STATIC_DIR / "viewer.html"
    if not viewer_html.exists():
        return {"error": "Viewer not available"}
    return FileResponse(viewer_html)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/metrics", dependencies=[Depends(require_auth)])
async def metrics():
    return get_metrics()
