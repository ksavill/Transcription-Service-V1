import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import asyncio
from contextlib import asynccontextmanager
import io

from config import get_value, load_config
from local_whisper_inference import (
    load_local_model,
    is_local_enabled,
    transcribe_with_local_concurrent
)
from openai_transcription import OpenAIInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed MIME types for audio files.
ALLOWED_MIMETYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"
}

# Allowed model names.
ALLOWED_MODELS = {
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "whisper-1",
    "local-whisper"
}

def is_openai_model(model_name: str) -> bool:
    """Return True if the model name is an OpenAI-based model."""
    return model_name in {"gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"}

# Instantiate our OpenAI transcription client once at the module level.
openai_client = OpenAIInterface()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config_lock = asyncio.Lock()
    # Load config on startup.
    await load_config(app)
    # Read the desired default primary model from config.
    openai_default = await get_value(app, "openai-default-transcribe-model")
    if not openai_default:
        openai_default = "gpt-4o-mini-transcribe"  # fallback if not set
    app.state.primary_model = openai_default

    # Optionally preload the local model.
    load_on_startup = await get_value(app, "load-local-whisper-on-startup")
    local_model_name = await get_value(app, "local-whisper-model")
    if load_on_startup and local_model_name:
        success = load_local_model(local_model_name)
        if success:
            logger.info("Local Whisper model '%s' loaded successfully.", local_model_name)
        else:
            logger.info("Failed to load local Whisper model '%s' on startup.", local_model_name)
    yield

app = FastAPI(
    title="Hybrid Transcription Service",
    description="Transcription service that uses OpenAI or local Whisper with fallback",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Mount static directory for index.html and styles.css.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html file at the root.
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/transcribe")
async def transcribe_audio(
    request: Request,  # to access app.state
    file: UploadFile = File(...),
    primary_model: Optional[str] = Form(None),  # use default if not provided
    allow_backup: bool = Form(True)
):
    """
    Accepts an audio file and parameters:
      - primary_model: one of ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1", "local-whisper"]
                       If not provided, uses the default from app.state.
      - allow_backup: if True, attempts fallback to the alternative method if the primary fails.
    """
    if not primary_model:
        primary_model = request.app.state.primary_model

    # Validate file MIME type.
    if file.content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type ({file.content_type}). Only WAV and MP3 are accepted."
        )

    if primary_model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{primary_model}' is not recognized or supported."
        )

    # Read file content.
    file_bytes = await file.read()

    # Decide routing based on primary_model.
    if is_openai_model(primary_model):
        # Wrap the bytes in a file-like object with a proper name.
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = file.filename

        if not openai_client.is_api_key_configured():
            if allow_backup:
                return await _attempt_local_fallback(file_bytes, file.filename)
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"OpenAI key is not configured; cannot use '{primary_model}'."
                )
        try:
            result = await openai_client.transcribe_audio(file_obj, model=primary_model)
            transcription_text = result.get("text", "")
            return {"transcription": transcription_text, "used_model": primary_model}
        except Exception as e:
            if allow_backup:
                return await _attempt_local_fallback(file_bytes, file.filename, error=str(e))
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"OpenAI transcription with '{primary_model}' failed. Error: {str(e)}"
                )
    else:
        # primary_model == "local-whisper"
        if not is_local_enabled():
            local_model_name = await get_value(request.app, "local-whisper-model")
            if not local_model_name:
                if allow_backup:
                    return await _attempt_openai_fallback(file_bytes, error="Local config missing.")
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="Local whisper model configuration is missing."
                    )
            success = load_local_model(local_model_name)
            if not success:
                if allow_backup:
                    return await _attempt_openai_fallback(file_bytes, error="Failed to load local model.")
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="Local whisper model is not available."
                    )
        try:
            local_result = await transcribe_with_local_concurrent(file_bytes, file.filename)
            return {"transcription": local_result, "used_model": "local-whisper"}
        except Exception as e:
            if allow_backup:
                return await _attempt_openai_fallback(file_bytes, error=str(e))
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Local inference failed. Error: {str(e)}"
                )

async def _attempt_local_fallback(file_bytes: bytes, filename: str, error: str = ""):
    """Fallback to local inference if OpenAI fails."""
    logger.info("Attempting fallback to local transcription. Reason: %s", error)
    if is_local_enabled():
        try:
            local_result = await transcribe_with_local_concurrent(file_bytes, filename)
            return {"transcription": local_result, "used_model": "local-whisper", "fallback_reason": error}
        except Exception as local_error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "All transcription methods failed. "
                    f"OpenAI error: {error}, Local inference error: {str(local_error)}"
                )
            )
    else:
        from config import get_value
        local_model_name = await get_value(app, "local-whisper-model")
        if local_model_name:
            success = load_local_model(local_model_name)
            if success:
                try:
                    local_result = await transcribe_with_local_concurrent(file_bytes, filename)
                    return {"transcription": local_result, "used_model": "local-whisper", "fallback_reason": error}
                except Exception as local_error:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "All transcription methods failed. "
                            f"OpenAI error: {error}, Local inference error: {str(local_error)}"
                        )
                    )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Primary transcription failed. Error: {error}. "
                "Local model is not available or could not be loaded."
            )
        )

async def _attempt_openai_fallback(file_bytes: bytes, error: str = ""):
    """Fallback to a default OpenAI model if local inference fails."""
    logger.info("Attempting fallback to OpenAI transcription. Reason: %s", error)
    if not openai_client.is_api_key_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Local inference failed: {error}. Cannot fall back to OpenAI, key is not configured."
            )
        )
    try:
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = "audiofile"
        result = await openai_client.transcribe_audio(file_obj, model="gpt-4o-transcribe")
        transcription_text = result.get("text", "")
        return {
            "transcription": transcription_text,
            "used_model": "gpt-4o-transcribe",
            "fallback_reason": error
        }
    except Exception as openai_error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"All transcription methods failed. Local error: {error}, "
                f"OpenAI error: {str(openai_error)}"
            )
        )

@app.post("/admin/enable-local-transcription")
async def enable_local_transcription():
    local_model_name = await get_value(app, "local-whisper-model")
    if not local_model_name:
        raise HTTPException(
            status_code=400,
            detail="Local whisper model configuration missing in config."
        )
    success = load_local_model(local_model_name)
    if success:
        return {"status": f"Local transcription has been enabled using model '{local_model_name}'."}
    else:
        raise HTTPException(
            status_code=400,
            detail="Failed to enable local transcription: Whisper module could not be loaded."
        )

@app.post("/admin/disable-local-transcription")
async def disable_local_transcription():
    from local_whisper_inference import is_local_enabled, disable_local_model
    if not is_local_enabled():
        return {"status": "Local transcription is already disabled."}
    disable_local_model()
    return {"status": "Local transcription has been disabled."}

@app.get("/available-models")
async def get_available_models():
    return [{"name": model} for model in ALLOWED_MODELS]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8070)