import asyncio
import load_local_whisper_model
import tempfile
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    import whisper
    whisper_available = True
except ImportError:
    whisper_available = False
    whisper = None

MAX_LOCAL_PROCESSES = 3
local_semaphore = asyncio.Semaphore(MAX_LOCAL_PROCESSES)
local_model = None
local_enabled = False

def load_local_model(model_name: str = "base") -> bool:
    global local_model, local_enabled
    if not whisper_available:
        print("Whisper is not installed or failed to import.")
        local_model = None
        local_enabled = False
        return False
    try:
        local_model = load_local_whisper_model.load_model(model_name)
        if local_model is None:
            return False
        local_enabled = True
        print(f"Local Whisper model '{model_name}' loaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to load local Whisper model '{model_name}'. Error: {e}")
        local_model = None
        local_enabled = False
        return False

def disable_local_model():
    global local_model, local_enabled
    local_model = None
    local_enabled = False

def is_local_enabled() -> bool:
    return local_enabled

def transcribe_with_local(file_bytes: bytes, file_name: str) -> str:
    # Determine file extension; default to .wav if none.
    _, ext = os.path.splitext(file_name)
    if not ext:
        ext = ".wav"
    # Create a temporary file without keeping it open.
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        temp_file.write(file_bytes)
        temp_file.close()  # close the file so that FFmpeg can open it on Windows
        result = local_model.transcribe(temp_file.name)
        logger.info("Transcription successful using model 'local-whisper'.")
    finally:
        os.remove(temp_file.name)
    return result.get("text", "")

async def transcribe_with_local_concurrent(file_bytes: bytes, file_name: str) -> str:
    async with local_semaphore:
        return await asyncio.to_thread(transcribe_with_local, file_bytes, file_name)