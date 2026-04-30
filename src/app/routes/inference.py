"""Inference HTTP routes — translation, TTS, and ASR/MT smoke checks."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.asr import transcribe_wav_path
from app.deps import require_auth
from app.languages import speech_languages, translation_languages
from app.mt import translate_texts
from app.scripts.asr_smoke import generate_silence_wav
from app.session_registry import registry

logger = logging.getLogger(__name__)

router = APIRouter()


# Soft caps kept well below MT_N_CTX (4096 tokens). Latin/Cyrillic/Arabic
# scripts produce ~0.3–0.7 tokens per char with TranslateGemma's tokenizer;
# CJK scripts produce ~1.5–3 tokens per char, so allow far fewer characters
# for those source languages or a long input would blow the context window.
TRANSLATE_TEXT_MAX_CHARS = 4000
TRANSLATE_TEXT_MAX_CHARS_DENSE = 1500
_DENSE_TOKEN_LANGS: frozenset[str] = frozenset({"zh", "ja", "ko"})


def _max_translate_chars(src_lang: str) -> int:
    return (
        TRANSLATE_TEXT_MAX_CHARS_DENSE
        if src_lang in _DENSE_TOKEN_LANGS
        else TRANSLATE_TEXT_MAX_CHARS
    )


class TranslateRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str


class TTSRequest(BaseModel):
    text: str
    lang: str


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------


@router.get("/api/languages")
async def api_languages(scope: str = "speech"):
    """Single source of truth for UI language dropdowns.

    Public — viewer.html is served without a session cookie, and the list
    itself isn't sensitive. Frontend pages call this on load and populate
    both their dropdowns and any local code→label maps from the result.

    scope=speech (default): foreign speech languages for index/viewer.
    scope=translate: speech languages plus German (translate page).
    """
    if scope == "translate":
        return translation_languages()
    return speech_languages()


# ---------------------------------------------------------------------------
# Smoke checks (auth-gated to keep them out of public reach)
# ---------------------------------------------------------------------------


@router.get("/asr_smoke", dependencies=[Depends(require_auth)])
async def asr_smoke():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        generate_silence_wav(wav_path, duration_sec=2.0)
        return transcribe_wav_path(wav_path)
    finally:
        os.unlink(wav_path)


@router.get("/mt_smoke", dependencies=[Depends(require_auth)])
async def mt_smoke():
    texts = ["Hello world!"]
    result = translate_texts(texts, src_lang="en", tgt_lang="de")
    return {"input": texts, "output": result}


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


@router.post("/api/translate", dependencies=[Depends(require_auth)])
async def api_translate(req: TranslateRequest):
    max_chars = _max_translate_chars(req.src_lang)
    if len(req.text) > max_chars:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Text zu lang ({len(req.text)} Zeichen, "
                f"Maximum {max_chars} für Quellsprache {req.src_lang!r})"
            ),
        )
    if not req.text.strip():
        return {"output": ""}
    if req.src_lang == req.tgt_lang:
        return {"output": req.text}
    try:
        output = await asyncio.to_thread(translate_texts, [req.text], req.src_lang, req.tgt_lang)
    except Exception:
        logger.exception("Text-to-text translation failed")
        raise HTTPException(status_code=500, detail="Übersetzung fehlgeschlagen") from None
    return {"output": output[0] if output else ""}


# ---------------------------------------------------------------------------
# TTS — auth-gated for hosts, token-gated for viewers.
# ---------------------------------------------------------------------------


@router.post("/api/tts", dependencies=[Depends(require_auth)])
async def tts_endpoint(request: TTSRequest):
    from app.tts import TTS_SUPPORTED_LANGS, synthesize_wav

    if request.lang not in TTS_SUPPORTED_LANGS:
        raise HTTPException(status_code=404, detail="Language not supported for TTS")
    audio_bytes = await asyncio.to_thread(synthesize_wav, request.text, request.lang)
    return Response(content=audio_bytes, media_type="audio/wav")


@router.post("/api/viewer/{token}/tts")
async def viewer_tts_endpoint(token: str, request: TTSRequest):
    from app.tts import TTS_SUPPORTED_LANGS, synthesize_wav

    if await registry.get(token) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if request.lang not in TTS_SUPPORTED_LANGS:
        raise HTTPException(status_code=404, detail="Language not supported for TTS")
    audio_bytes = await asyncio.to_thread(synthesize_wav, request.text, request.lang)
    return Response(content=audio_bytes, media_type="audio/wav")


# ---------------------------------------------------------------------------
# Batch transcription + translation (file upload)
# ---------------------------------------------------------------------------


@router.post("/transcribe_translate", dependencies=[Depends(require_auth)])
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
