from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

from timeout_function_decorator import timeout
from zyphra import ZyphraClient

from utts.config import MAXHITS, TIMEOUT, get_settings
from utts.utils import convert_to_enum


class Voice(str, Enum):
    """Available default voices for Zyphra TTS API."""

    AMERICAN_FEMALE = "american_female"
    AMERICAN_MALE = "american_male"
    ANIME_GIRL = "anime_girl"
    BRITISH_FEMALE = "british_female"
    BRITISH_MALE = "british_male"
    ENERGETIC_BOY = "energetic_boy"
    ENERGETIC_GIRL = "energetic_girl"
    JAPANESE_FEMALE = "japanese_female"
    JAPANESE_MALE = "japanese_male"


class Model(str, Enum):
    """Available models for Zyphra TTS API."""

    ZONOS_TRANSFORMER = "zonos-v0.1-transformer"
    ZONOS_HYBRID = "zonos-v0.1-hybrid"


class Language(str, Enum):
    """Available languages for Zyphra TTS API."""

    ENGLISH_US = "en-us"
    FRENCH = "fr-fr"
    GERMAN = "de"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "cmn"


class MimeType(str, Enum):
    """Available output formats for Zyphra TTS API."""

    WEBM = "audio/webm"
    OGG = "audio/ogg"
    WAV = "audio/wav"
    MP3 = "audio/mp3"
    MP4 = "audio/mp4"
    AAC = "audio/aac"


@lru_cache(MAXHITS)
def get_client() -> ZyphraClient:
    """Returns a Zyphra client."""
    settings = get_settings().zyphra
    assert settings is not None, "Zyphra settings are not configured"
    return ZyphraClient(api_key=settings.api_key)


@timeout(TIMEOUT)
def generate(
    text: str,
    voice: Union[Voice, str] = Voice.AMERICAN_FEMALE,
    model: Union[Model, str] = Model.ZONOS_TRANSFORMER,
    language: Union[Language, str] = Language.ENGLISH_US,
    speaking_rate: float = 15.0,
    pitch_std: Optional[float] = None,
    mime_type: Union[MimeType, str] = MimeType.WEBM,
    speaker_audio: Optional[str] = None,
    voice_name: Optional[str] = None,
    fmax: Optional[float] = None,
    emotion: Optional[Dict[str, float]] = None,
    speaker_noised: Optional[bool] = None,
    vqscore: Optional[float] = None,
) -> bytes:
    """
    Generates audio from text using Zyphra TTS API (synchronous version).

    Args:
        text: Text to convert to speech
        voice: Default voice to use (american_female, american_male, etc.)
        model: TTS model to use (zonos-v0.1-transformer, zonos-v0.1-hybrid)
        language: Language code (en-us, fr-fr, etc.)
        speaking_rate: Speaking rate (5-35, default: 15.0)
        pitch_std: Pitch standard deviation (0-500, default: 45.0, transformer model only)
        mime_type: Output audio format (audio/webm, audio/mp3, etc.)
        speaker_audio: Base64 encoded audio for voice cloning
        voice_name: Name of a custom voice to use
        fmax: Maximum frequency for audio generation (default: 22050)
        emotion: Emotional weights for speech generation (transformer model only)
        speaker_noised: Denoises reference audio to improve voice stability (hybrid model only)
        vqscore: Controls voice quality vs. speaker similarity (0.6-0.8)

    Returns:
        Audio data as bytes
    """
    voice_enum = convert_to_enum(Voice, voice)
    model_enum = convert_to_enum(Model, model)
    language_enum = convert_to_enum(Language, language)
    mime_type_enum = convert_to_enum(MimeType, mime_type)

    client = get_client()

    params: Dict[str, Any] = {
        "text": text,
        "model": model_enum.value,
        "language_iso_code": language_enum.value,
        "speaking_rate": speaking_rate,
        "mime_type": mime_type_enum.value,
    }

    # Add optional parameters only if they are provided
    if pitch_std is not None and model_enum == Model.ZONOS_TRANSFORMER:
        params["pitch_std"] = pitch_std

    if fmax is not None:
        params["fmax"] = fmax

    if emotion is not None and model_enum == Model.ZONOS_TRANSFORMER:
        params["emotion"] = emotion

    if speaker_noised is not None and model_enum == Model.ZONOS_HYBRID:
        params["speaker_noised"] = speaker_noised

    if vqscore is not None:
        params["vqscore"] = vqscore

    # Voice selection priority: voice_name > speaker_audio > default_voice_name
    if voice_name is not None:
        params["voice_name"] = voice_name
    elif speaker_audio is not None:
        params["speaker_audio"] = speaker_audio
    else:
        params["default_voice_name"] = voice_enum.value

    # Get binary audio data from the API response
    response = client.audio.speech.create(**params)
    if isinstance(response, Path):
        return response.read_bytes()
    return response


@timeout(TIMEOUT)
async def agenerate(
    text: str,
    voice: Union[Voice, str] = Voice.AMERICAN_FEMALE,
    model: Union[Model, str] = Model.ZONOS_TRANSFORMER,
    language: Union[Language, str] = Language.ENGLISH_US,
    speaking_rate: float = 15.0,
    pitch_std: Optional[float] = None,
    mime_type: Union[MimeType, str] = MimeType.WEBM,
    speaker_audio: Optional[str] = None,
    voice_name: Optional[str] = None,
    fmax: Optional[float] = None,
    emotion: Optional[Dict[str, float]] = None,
    speaker_noised: Optional[bool] = None,
    vqscore: Optional[float] = None,
) -> bytes:
    """
    Generates audio from text using Zyphra TTS API asynchronously.

    Args:
        text: Text to convert to speech
        voice: Default voice to use (american_female, american_male, etc.)
        model: TTS model to use (zonos-v0.1-transformer, zonos-v0.1-hybrid)
        language: Language code (en-us, fr-fr, etc.)
        speaking_rate: Speaking rate (5-35, default: 15.0)
        pitch_std: Pitch standard deviation (0-500, default: 45.0, transformer model only)
        mime_type: Output audio format (audio/webm, audio/mp3, etc.)
        speaker_audio: Base64 encoded audio for voice cloning
        voice_name: Name of a custom voice to use
        fmax: Maximum frequency for audio generation (default: 22050)
        emotion: Emotional weights for speech generation (transformer model only)
        speaker_noised: Denoises reference audio to improve voice stability (hybrid model only)
        vqscore: Controls voice quality vs. speaker similarity (0.6-0.8)

    Returns:
        Audio data as bytes
    """
    # For now, this is just a wrapper around the synchronous function
    # In the future, this could be updated to use async features
    return generate(
        text=text,
        voice=voice,
        model=model,
        language=language,
        speaking_rate=speaking_rate,
        pitch_std=pitch_std,
        mime_type=mime_type,
        speaker_audio=speaker_audio,
        voice_name=voice_name,
        fmax=fmax,
        emotion=emotion,
        speaker_noised=speaker_noised,
        vqscore=vqscore,
    )
