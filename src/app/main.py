"""
FastAPI application for LinguaGap real-time transcription and translation.

This is the main entry point for the application. It provides:
    - HTTP endpoints for health checks, metrics, and file upload transcription
    - WebSocket endpoints for real-time streaming transcription
    - Static file serving for the web UI
    - Demo authentication for the desktop interface

Endpoints:
    GET  /              - Web interface (requires login)
    GET  /login         - Login page
    POST /api/login     - Login endpoint
    POST /api/logout    - Logout endpoint
    GET  /api/me        - Current user info
    GET  /health        - Health check
    GET  /metrics       - Performance metrics (requires login)
    POST /transcribe_translate - File upload transcription (requires login)
    POST /api/tts       - Text-to-speech (requires login)
    WS   /ws            - Real-time streaming WebSocket (requires login)
    GET  /viewer/{token} - Mobile viewer page (public)
    WS   /ws/viewer/{token} - Read-only viewer WebSocket (public)

Startup:
    All models are warmed up on startup via the lifespan handler to minimize
    cold-start latency on first request.
"""

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from app.asr import transcribe_wav_path
from app.auth import get_current_user, verify_credentials
from app.backends import get_asr_backend, get_summarization_backend, get_translation_backend
from app.mt import translate_texts
from app.scripts.asr_smoke import generate_silence_wav
from app.streaming import get_metrics, handle_viewer_websocket, handle_websocket

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

SESSION_SECRET = os.getenv("SESSION_SECRET", "linguagap-dev-secret-change-me")


def warmup_models():
    """
    Warm up all ML models on startup.

    This runs during application startup to ensure all models are loaded
    and ready before the first request. Without warmup, the first request
    would experience significant latency (minutes for large models).

    Models warmed up:
        - ASR (via backend): ~2-3 GB VRAM
        - Translation (via backend): ~8 GB VRAM
        - Summarization (optional, via backend): ~4 GB VRAM
        - TTS (optional, KugelAudio 4-bit): ~4 GB VRAM

    Total warmup time is typically 5-10 minutes depending on network speed
    for model downloads and GPU initialization.
    """
    logger.info("Warming up ASR backend...")
    asr = get_asr_backend()
    asr.warmup()
    logger.info("ASR backend ready")

    logger.info("Warming up MT backend...")
    mt = get_translation_backend()
    mt.warmup()
    logger.info("MT backend ready")

    summ = get_summarization_backend()
    if summ is not None:
        logger.info("Warming up summarization backend...")
        summ.warmup()
        logger.info("Summarization backend ready")

    if os.getenv("TTS_BACKEND"):
        logger.info("Warming up TTS model...")
        from app.tts import get_tts_model

        get_tts_model()
        logger.info("TTS model ready")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    warmup_models()
    yield


app = FastAPI(
    title="LinguaGap",
    description="Real-time speech transcription and translation",
    lifespan=lifespan,
)

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, session_cookie="linguagap_session")

STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def require_auth(request: Request):
    if not request.session.get("username"):
        raise HTTPException(status_code=401, detail="Not authenticated")


# ---------------------------------------------------------------------------
# Auth routes (public)
# ---------------------------------------------------------------------------


@app.get("/login")
async def login_page():
    return FileResponse(STATIC_DIR / "login.html")


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/login")
async def api_login(request: Request, body: LoginRequest):
    account = verify_credentials(body.username, body.password)
    if account is None:
        return JSONResponse({"error": "Invalid credentials"}, status_code=401)
    request.session["username"] = account.username
    request.session["display_name"] = account.display_name
    request.session["logo_url"] = account.logo_url
    return {"ok": True, "display_name": account.display_name, "logo_url": account.logo_url}


@app.post("/api/logout")
async def api_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
async def api_me(request: Request):
    user = get_current_user(request)
    if user is None:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    return user


# ---------------------------------------------------------------------------
# Protected routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root(request: Request):
    if not request.session.get("username"):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics", dependencies=[Depends(require_auth)])
async def metrics():
    return get_metrics()


@app.get("/asr_smoke", dependencies=[Depends(require_auth)])
async def asr_smoke():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        generate_silence_wav(wav_path, duration_sec=2.0)
        return transcribe_wav_path(wav_path)
    finally:
        os.unlink(wav_path)


@app.get("/mt_smoke", dependencies=[Depends(require_auth)])
async def mt_smoke():
    texts = ["Hello world!"]
    result = translate_texts(texts, src_lang="en", tgt_lang="de")
    return {"input": texts, "output": result}


class TTSRequest(BaseModel):
    text: str
    lang: str


@app.post("/api/tts", dependencies=[Depends(require_auth)])
async def tts_endpoint(request: TTSRequest):
    from app.tts import TTS_SUPPORTED_LANGS, synthesize_wav

    if request.lang not in TTS_SUPPORTED_LANGS:
        raise HTTPException(status_code=404, detail="Language not supported for TTS")
    audio_bytes = await asyncio.to_thread(synthesize_wav, request.text, request.lang)
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/transcribe_translate", dependencies=[Depends(require_auth)])
async def transcribe_translate(
    file: UploadFile = File(...),
    src_lang: str = Form("auto"),
):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        content = await file.read()
        f.write(content)
        audio_path = f.name

    try:
        asr_result = transcribe_wav_path(audio_path)

        detected_lang = asr_result["language"]
        if src_lang == "auto":
            src_lang = detected_lang

        segments = []
        for i, seg in enumerate(asr_result["segments"]):
            src_text = seg["text"].strip()
            if src_text:
                de_text = translate_texts([src_text], src_lang=src_lang, tgt_lang="de")[0]
            else:
                de_text = ""

            segments.append(
                {
                    "id": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "src": src_text,
                    "de": de_text,
                }
            )

        return {
            "src_lang_detected": detected_lang,
            "segments": segments,
        }
    finally:
        os.unlink(audio_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session = websocket.scope.get("session", {})
    if not session.get("username"):
        await websocket.close(code=4001, reason="Not authenticated")
        return
    await handle_websocket(websocket)


@app.get("/viewer/{token}")
async def viewer_page(token: str):  # noqa: ARG001
    """Serve the mobile viewer page.

    The token is validated client-side via WebSocket connection.
    """
    viewer_html = STATIC_DIR / "viewer.html"
    if not viewer_html.exists():
        return {"error": "Viewer not available"}
    return FileResponse(viewer_html)


@app.websocket("/ws/viewer/{token}")
async def viewer_websocket_endpoint(websocket: WebSocket, token: str):
    """WebSocket endpoint for read-only viewers."""
    await handle_viewer_websocket(websocket, token)
