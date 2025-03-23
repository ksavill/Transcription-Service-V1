import os
import io
import aiohttp
import aiofiles
from typing import Union, IO, Optional
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OpenAIInterface:
    def __init__(self, default_model: str = "got-4o-mini-transcribe", api_key: Optional[str] = None):
        self.default_model = default_model
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get('openai_token') or os.environ.get('OPENAI_API_KEY')
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}"
                }
            )
        return self._session

    def is_api_key_configured(self) -> bool:
        return bool(self.api_key)

    async def transcribe_audio(
        self,
        file: Union[str, IO],
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0
    ) -> dict:
        """
        Transcribe an audio file using the specified transcription model.
        
        Supported models: "whisper-1", "gpt-4o-mini-transcribe", and "gpt-4o-transcribe".

        :param file: Either a file path (str) or a file‑like object (opened in binary mode).
        :param model: Transcription model to use. If None, defaults to self.default_model.
        :param language: Optional language code (e.g., "en").
        :param prompt: Optional prompt to guide transcription.
        :param response_format: Output format (for newer GPT-4o models, only "json" or "text" are supported).
        :param temperature: Temperature for generation (default is 0.0).
        :return: A dict containing the transcription result.
        """
        selected_model = model if model is not None else self.default_model

        # Prepare a file‑like object.
        if isinstance(file, str):
            async with aiofiles.open(file, "rb") as f:
                file_bytes = await f.read()
            file_obj = io.BytesIO(file_bytes)
            file_obj.name = os.path.basename(file)
        else:
            file_obj = file
            if not hasattr(file_obj, "name"):
                file_obj.name = "audiofile"

        data = aiohttp.FormData()
        data.add_field("file", file_obj, filename=file_obj.name, content_type="audio/mpeg")
        data.add_field("model", selected_model)
        if language:
            data.add_field("language", language)
        if prompt:
            data.add_field("prompt", prompt)
        data.add_field("response_format", response_format)
        data.add_field("temperature", str(temperature))

        session = await self._get_session()
        url = "https://api.openai.com/v1/audio/transcriptions"

        try:
            async with session.post(url, data=data) as resp:
                resp.raise_for_status()
                result = await resp.json()
                logger.info("Transcription successful using model '%s'.", selected_model)
                return result
        except aiohttp.ClientResponseError as e:
            logger.error("API request failed: %s", e)
            raise

    async def transcribe_with_whisper(
        self,
        file: Union[str, IO],
        **kwargs
    ) -> dict:
        """
        Convenience method to transcribe audio using the whisper-1 model.
        """
        return await self.transcribe_audio(file, model="whisper-1", **kwargs)