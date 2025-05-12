from enum import Enum
from functools import lru_cache

from openai import AsyncOpenAI, OpenAI
from timeout_function_decorator import timeout

from utts.config import MAXHITS, TIMEOUT, get_settings
from utts.utils import convert_to_enum


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


@lru_cache(MAXHITS)
def get_client() -> OpenAI:
    """Returns a synchronous OpenAI client."""
    settings = get_settings().openai
    assert settings is not None, "OpenAI settings are not configured"
    return OpenAI(api_key=settings.api_key, organization=settings.organization_id)


@lru_cache(MAXHITS)
def get_aclient() -> AsyncOpenAI:
    """Returns an asynchronous OpenAI client."""
    settings = get_settings().openai
    assert settings is not None, "OpenAI settings are not configured"
    return AsyncOpenAI(api_key=settings.api_key, organization=settings.organization_id)


@timeout(TIMEOUT)
def generate(text: str, voice: Voice | str = Voice.ALLOY, model: Model | str = Model.TTS_1) -> bytes:
    """
    Generates audio from text using OpenAI TTS API (synchronous version).

    Args:
        text: Text to convert to speech
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        model: TTS model to use (tts-1, tts-1-hd)

    Returns:
        Audio data as bytes
    """
    voice = convert_to_enum(Voice, voice)
    model = convert_to_enum(Model, model)
    client = get_client()
    response = client.audio.speech.create(model=model, voice=voice, input=text)
    return response.content


@timeout(TIMEOUT)
async def agenerate(text: str, voice: Voice | str = Voice.ALLOY, model: Model | str = Model.TTS_1) -> bytes:
    """
    Generates audio from text using OpenAI TTS API asynchronously.

    Args:
        text: Text to convert to speech
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        model: TTS model to use (tts-1, tts-1-hd)

    Returns:
        Audio data as bytes
    """
    voice = convert_to_enum(Voice, voice)
    model = convert_to_enum(Model, model)
    client = get_aclient()
    response = await client.audio.speech.create(model=model, voice=voice, input=text)
    return response.content
