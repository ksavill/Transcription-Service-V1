from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import Optional
import uvicorn
import os
import httpx
import asyncio
from tempfile import NamedTemporaryFile

from load_local_whisper_model import load_model

# Global configuration variables
MAX_LOCAL_PROCESSES = 3                # Maximum concurrent local transcriptions
PREFERRED_PROCESSING = "local"         # Preferred method: "local" or "openai"
local_semaphore = asyncio.Semaphore(MAX_LOCAL_PROCESSES)
LOCAL_MODEL = "base"                   # Set local whisper model

# Attempt to import the whisper module and load the local model on startup.
load_success = load_model(LOCAL_MODEL)
if load_success:
    local_enabled = True
else:
    local_enabled = False

app = FastAPI()

# Allowed MIME types for audio files (wav and mp3)
ALLOWED_MIMETYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"}
ALLOWED_PROCESSING_METHODS = {"local", "openai"}

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("openai_token")

async def transcribe_with_openai(file_bytes: bytes, filename: str, content_type: str, api_token: str) -> str:
    """
    Transcribes the given audio file using OpenAI's Whisper API.
    """
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_token}"}
    data = {"model": "whisper-1"}
    files = {"file": (filename, file_bytes, content_type)}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data, files=files)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="OpenAI transcription failed")
    
    result = response.json()
    return result.get("text", "")

def transcribe_with_local(file_bytes: bytes, file_filename: str) -> str:
    """
    Transcribes the given audio file using the local Whisper model.
    Writes the file bytes to a temporary file and calls the model's transcribe method.
    """
    # Extract file extension from filename; default to .wav if not present.
    _, ext = os.path.splitext(file_filename)
    if not ext:
        ext = ".wav"
    
    with NamedTemporaryFile(suffix=ext, delete=True) as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()  # Ensure all bytes are written to disk
        # Transcribe using the loaded local model
        result = local_model.transcribe(temp_file.name)
    
    return result.get("text", "")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...), 
    processing_method: Optional[str] = Form(None)
):
    # Validate file MIME type
    if file.content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only WAV and MP3 files are accepted."
        )
    
    # Read file content as bytes
    file_bytes = await file.read()

    # Automatic processing method determination if none is provided
    if processing_method is None:
        if PREFERRED_PROCESSING == "local":
            if local_enabled:
                # Check if there's available capacity for local processing
                if local_semaphore._value > 0:  # using _value as a prototype check
                    processing_method = "local"
                elif OPENAI_API_KEY:
                    processing_method = "openai"
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="Local transcription capacity exceeded and no fallback is available. Please try again later."
                    )
            elif OPENAI_API_KEY:
                processing_method = "openai"
            else:
                raise HTTPException(
                    status_code=503,
                    detail="No transcription methods are currently available."
                )
        else:  # Preferred processing is "openai"
            if OPENAI_API_KEY:
                processing_method = "openai"
            elif local_enabled and local_semaphore._value > 0:
                processing_method = "local"
            else:
                raise HTTPException(
                    status_code=503,
                    detail="No transcription methods are currently available."
                )
    else:
        # Validate explicit processing_method
        if processing_method not in ALLOWED_PROCESSING_METHODS:
            raise HTTPException(
                status_code=400,
                detail="Invalid processing_method. Must be 'local' or 'openai'."
            )
        if processing_method == "local" and not local_enabled:
            raise HTTPException(
                status_code=400,
                detail="Local transcription is not available because the whisper module is not installed."
            )

    # Process the transcription based on the chosen method
    if processing_method == "openai":
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=400, detail="OpenAI API key not configured.")
        transcription_result = await transcribe_with_openai(file_bytes, file.filename, file.content_type, OPENAI_API_KEY)
    elif processing_method == "local":
        # Acquire a semaphore permit to limit concurrent local transcriptions
        async with local_semaphore:
            transcription_result = await asyncio.to_thread(transcribe_with_local, file_bytes, file.filename)

    return {"transcription": transcription_result, "used_method": processing_method}

@app.post("/admin/enable-local-transcription")
async def enable_local_transcription():
    """
    Admin endpoint to attempt importing and initializing the local Whisper model.
    If successful, updates the global flag to enable local transcription.
    """
    global local_enabled, local_model
    if local_enabled:
        return {"status": "Local transcription already enabled."}
    
    load_success = load_model(LOCAL_MODEL)
    if load_success:
        local_enabled = True
        return {"status": "Local transcription has been enabled."}
    else:
        raise HTTPException (
            status_code=400,
            detail=f"Failed to enable local transcription: Whisper module could not be loaded."
        )

@app.post("/admin/disable-local-transcription")
async def disable_local_transcription():
    """
    Admin endpoint to disable local transcription by updating the global flag.
    This effectively prevents using the local Whisper model.
    """
    global local_enabled, local_model
    if not local_enabled:
        return {"status": "Local transcription is already disabled."}
    
    local_enabled = False
    local_model = None
    return {"status": "Local transcription has been disabled."}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8070, reload=True)