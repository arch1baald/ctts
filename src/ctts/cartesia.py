import wave
from enum import Enum
from functools import lru_cache
from io import BytesIO
from typing import BinaryIO, List, Optional, Union

import numpy as np
from cartesia import AsyncCartesia, Cartesia
from pydantic import BaseModel

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
    duration: Optional[float] = None,
) -> bytes:
    """
    Generates audio from text using Cartesia TTS API.

    Args:
        text: Text to convert to speech
        model: TTS model to use (sonic-2, sonic-turbo, sonic)
        language: Language code (en, fr, de, etc.)
        voice_id: ID of a saved voice to use
        voice_audio: Audio sample for voice cloning
        duration: Target duration in seconds for the generated audio

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

    # Add duration parameter if specified
    if duration is not None:
        params["duration"] = duration

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
    duration: Optional[float] = None,
) -> bytes:
    """
    Asynchronously generates audio from text using Cartesia TTS API.

    Args:
        text: Text to convert to speech
        model: TTS model to use (sonic-2, sonic-turbo, sonic)
        language: Language code (en, fr, de, etc.)
        voice_id: ID of a saved voice to use
        voice_audio: Audio sample for voice cloning
        duration: Target duration in seconds for the generated audio

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

    # Add duration parameter if specified
    if duration is not None:
        params["duration"] = duration

    # Collect bytes from async iterator
    output = BytesIO()
    async for chunk in client.tts.bytes(**params):
        output.write(chunk)

    return output.getvalue()


def _wrap_pcm_f32_to_wav(raw_data: bytes, sample_rate: int = 44100) -> bytes:
    """
    Convert raw PCM float32 data to WAV with PCM S16LE encoding.
    """
    # Interpret raw data as float32
    floats = np.frombuffer(raw_data, dtype=np.float32)
    # Scale to int16 range
    ints = np.clip(floats * 32767.0, -32768, 32767).astype(np.int16)
    # Write WAV
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for PCM 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(ints.tobytes())
    return buf.getvalue()


class TTSWithTimestampsResponse(BaseModel):
    """
    Pydantic model for TTS response with word and phoneme timestamps.
    """

    audio: bytes  # WAV bytes
    words: List[str]
    word_starts: List[float]
    word_ends: List[float]
    phonemes: List[str]
    phoneme_starts: List[float]
    phoneme_ends: List[float]


async def agenerate_with_timestamps(
    text: str,
    model: Union[Model, str] = Model.SONIC_2,
    language: Union[Language, str] = Language.ENGLISH,
    voice_id: Optional[str] = None,
    voice_audio: Optional[Union[bytes, BinaryIO]] = None,
) -> TTSWithTimestampsResponse:
    """
    Asynchronously generates audio and returns a Pydantic object with WAV bytes,
    word-level and phoneme-level timestamps.
    """
    client = get_async_client()
    model_enum = convert_to_enum(Model, model)
    language_enum = convert_to_enum(Language, language)

    ws = await client.tts.websocket()
    params = {
        "model_id": model_enum.value,
        "transcript": text,
        "language": language_enum.value,
        "voice": {"id": voice_id}
        if voice_id
        else ({"audio": voice_audio} if voice_audio else {"id": DEFAULT_VOICE_ID}),
        "output_format": {"container": "raw", "encoding": "pcm_f32le", "sample_rate": 44100},
        "add_timestamps": True,
        "add_phoneme_timestamps": True,
        "stream": True,
    }

    stream = await ws.send(**params)
    audio_chunks: List[bytes] = []
    words: List[str] = []
    word_starts: List[float] = []
    word_ends: List[float] = []
    phonemes: List[str] = []
    phoneme_starts: List[float] = []
    phoneme_ends: List[float] = []

    async for out in stream:  # type: ignore
        if out.audio:
            audio_chunks.append(out.audio)
        if out.word_timestamps:
            words.extend(out.word_timestamps.words)
            word_starts.extend(out.word_timestamps.start)
            word_ends.extend(out.word_timestamps.end)
        if hasattr(out, "phoneme_timestamps") and out.phoneme_timestamps:
            phonemes.extend(out.phoneme_timestamps.phonemes)
            phoneme_starts.extend(out.phoneme_timestamps.start)
            phoneme_ends.extend(out.phoneme_timestamps.end)

    await ws.close()
    raw = b"".join(audio_chunks)
    wav = _wrap_pcm_f32_to_wav(raw)

    return TTSWithTimestampsResponse(
        audio=wav,
        words=words,
        word_starts=word_starts,
        word_ends=word_ends,
        phonemes=phonemes,
        phoneme_starts=phoneme_starts,
        phoneme_ends=phoneme_ends,
    )
