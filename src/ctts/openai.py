from functools import lru_cache
from enum import Enum

from openai import OpenAI

from ctts.config import get_settings
from ctts.utils import convert_to_enum

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


@lru_cache()
def get_openai_client() -> OpenAI:
    """Returns an OpenAI client."""
    settings = get_settings().openai
    assert settings is not None, "OpenAI settings are not configured"
    return OpenAI(api_key=settings.api_key, organization=settings.organization_id)


def generate(
    text: str, 
    voice: Voice | str = Voice.ALLOY, 
    model: Model | str = Model.TTS_1
) -> bytes:
    """
    Generates audio from text using OpenAI TTS API.
    
    Args:
        text: Text to convert to speech
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        model: TTS model to use (tts-1, tts-1-hd)
        
    Returns:
        Audio data as bytes
    """
    voice = convert_to_enum(Voice, voice)
    model = convert_to_enum(Model, model)
    client = get_openai_client()
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    return response.content
