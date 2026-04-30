"""HTTP route modules.

Each submodule exposes a `router` (FastAPI APIRouter) that the application
root in `app.main` mounts via `app.include_router(...)`. WebSocket
endpoints stay in `app.main` directly because they don't compose well
with APIRouter for our setup.
"""
