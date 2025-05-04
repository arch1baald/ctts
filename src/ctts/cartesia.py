from enum import Enum
from functools import lru_cache
from io import BytesIO
from typing import BinaryIO, Optional, Union

from cartesia import AsyncCartesia, Cartesia

from ctts.config import get_settings
from ctts.utils import convert_to_enum


class Model(str, Enum):
    """Available models for Cartesia TTS API."""

    # Base models
    SONIC_2 = "sonic-2"
    SONIC_TURBO = "sonic-turbo"
    SONIC = "sonic"

    # Specific snapshots
    SONIC_2_2025_04_16 = "sonic-2-2025-04-16"
    SONIC_2_2025_03_07 = "sonic-2-2025-03-07"
    SONIC_TURBO_2025_03_07 = "sonic-turbo-2025-03-07"
    SONIC_2024_12_12 = "sonic-2024-12-12"
    SONIC_2024_10_19 = "sonic-2024-10-19"


class Language(str, Enum):
    """Available languages for Cartesia TTS API."""

    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    HINDI = "hi"
    ITALIAN = "it"
    KOREAN = "ko"
    DUTCH = "nl"
    POLISH = "pl"
    RUSSIAN = "ru"
    SWEDISH = "sv"
    TURKISH = "tr"


# Default voice ID from Cartesia documentation
DEFAULT_VOICE_ID = "694f9389-aac1-45b6-b726-9d9369183238"


@lru_cache()
def get_client() -> Cartesia:
    """Returns a Cartesia client."""

    settings = get_settings().cartesia
    assert settings is not None, "Cartesia settings are not configured"

    return Cartesia(api_key=settings.api_key)


@lru_cache()
def get_async_client() -> AsyncCartesia:
    """Returns an async Cartesia client."""

    settings = get_settings().cartesia
    assert settings is not None, "Cartesia settings are not configured"

    return AsyncCartesia(api_key=settings.api_key)


def generate(
    text: str,
    model: Union[Model, str] = Model.SONIC_2,
    language: Union[Language, str] = Language.ENGLISH,
    voice_id: Optional[str] = None,
    voice_audio: Optional[Union[bytes, BinaryIO]] = None,
) -> bytes:
    """
    Generates audio from text using Cartesia TTS API.

    Args:
        text: Text to convert to speech
        model: TTS model to use (sonic-2, sonic-turbo, sonic)
        language: Language code (en, fr, de, etc.)
        voice_id: ID of a saved voice to use
        voice_audio: Audio sample for voice cloning

    Returns:
        Audio data as bytes
    """
    client = get_client()
    model_enum = convert_to_enum(Model, model)
    language_enum = convert_to_enum(Language, language)

    params = {
        "model_id": model_enum.value,
        "transcript": text,
        "language": language_enum.value,
        "output_format": {
            "container": "wav",
            "sample_rate": 44100,
            "encoding": "pcm_f32le",
        },
    }

    # Add voice - required parameter
    if voice_id:
        params["voice"] = {"id": voice_id}
    elif voice_audio:
        params["voice"] = {"audio": voice_audio}
    else:
        # Use default voice if not specified
        params["voice"] = {"id": DEFAULT_VOICE_ID}

    # Collect bytes from iterator
    output = BytesIO()
    for chunk in client.tts.bytes(**params):
        output.write(chunk)

    return output.getvalue()


async def agenerate(
    text: str,
    model: Union[Model, str] = Model.SONIC_2,
    language: Union[Language, str] = Language.ENGLISH,
    voice_id: Optional[str] = None,
    voice_audio: Optional[Union[bytes, BinaryIO]] = None,
) -> bytes:
    """
    Asynchronously generates audio from text using Cartesia TTS API.

    Args:
        text: Text to convert to speech
        model: TTS model to use (sonic-2, sonic-turbo, sonic)
        language: Language code (en, fr, de, etc.)
        voice_id: ID of a saved voice to use
        voice_audio: Audio sample for voice cloning

    Returns:
        Audio data as bytes
    """
    client = get_async_client()
    model_enum = convert_to_enum(Model, model)
    language_enum = convert_to_enum(Language, language)

    params = {
        "model_id": model_enum.value,
        "transcript": text,
        "language": language_enum.value,
        "output_format": {
            "container": "wav",
            "sample_rate": 44100,
            "encoding": "pcm_f32le",
        },
    }

    # Add voice - required parameter
    if voice_id:
        params["voice"] = {"id": voice_id}
    elif voice_audio:
        params["voice"] = {"audio": voice_audio}
    else:
        # Use default voice if not specified
        params["voice"] = {"id": DEFAULT_VOICE_ID}

    # Collect bytes from async iterator
    output = BytesIO()
    async for chunk in client.tts.bytes(**params):
        output.write(chunk)

    return output.getvalue()
