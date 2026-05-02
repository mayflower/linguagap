"""Shared FastAPI dependencies and constants.

Living between the route modules and the application root so each router
can pull in `require_auth` / `require_admin` and the static-files path
without re-importing through main.py (which would create a cycle).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request

# Repository root → static/. Resolved once at import time so request paths
# stay anchored regardless of the working directory.
STATIC_DIR = Path(__file__).parent.parent.parent / "static"


def require_auth(request: Request) -> None:
    """FastAPI dependency: 401 unless the session has an authenticated user."""
    if not request.session.get("email"):
        raise HTTPException(status_code=401, detail="Not authenticated")


def require_admin(request: Request) -> None:
    """FastAPI dependency: 403 unless the session is flagged as admin."""
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
