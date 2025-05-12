from enum import Enum
from typing import cast

from openai import OpenAI
from timeout_function_decorator import timeout

from .base import ProviderClient


class Voice(str, Enum):
    """Available voices for OpenAI TTS API."""

    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


class Model(str, Enum):
    """Available models for OpenAI TTS API."""

    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"


class OpenAIClient(ProviderClient):
    """OpenAI text-to-speech API client."""

    def __init__(self, api_key: str, timeout: float):
        self.api_key = api_key
        self.timeout = timeout
        self.client = self.get_client(api_key)

    def get_client(self, api_key: str) -> OpenAI:
        return OpenAI(api_key=api_key)

    def generate(self, text: str, voice: Voice | str = Voice.ALLOY, model: Model | str = Model.TTS_1) -> bytes:
        """
        Generates audio from text using OpenAI TTS API.

        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model to use (tts-1, tts-1-hd)

        Returns:
            Audio data as bytes
        """
        timed_func = timeout(self.timeout)(self._generate)
        return cast(bytes, timed_func(text, voice, model))

    def _generate(self, text: str, voice: Voice | str = Voice.ALLOY, model: Model | str = Model.TTS_1) -> bytes:
        response = self.client.audio.speech.create(model=model, voice=voice, input=text)
        return response.content
