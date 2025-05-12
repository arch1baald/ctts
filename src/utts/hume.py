import base64
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional, Union

from hume import AsyncHumeClient, HumeClient
from hume.tts import FormatPcm, PostedContextWithGenerationId, PostedUtterance, PostedUtteranceVoiceWithName
from timeout_function_decorator import timeout

from utts.config import MAXHITS, TIMEOUT, get_settings
from utts.utils import convert_to_enum


class Format(str, Enum):
    """Available output formats for Hume TTS API."""

    WAV = "wav"
    PCM = "pcm"


class EmotionPreset(str, Enum):
    """Predefined emotion presets for Hume TTS API."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"


@lru_cache(MAXHITS)
def get_client() -> HumeClient:
    """Returns a Hume client."""
    settings = get_settings().hume
    assert settings is not None, "Hume settings are not configured"
    return HumeClient(api_key=settings.api_key)


@lru_cache(MAXHITS)
def get_aclient() -> AsyncHumeClient:
    """Returns a Hume client."""
    settings = get_settings().hume
    assert settings is not None, "Hume settings are not configured"
    return AsyncHumeClient(api_key=settings.api_key)


@timeout(TIMEOUT)
def generate(
    text: str,
    description: Optional[str] = None,
    voice: Optional[str] = None,
    context_generation_id: Optional[str] = None,
    format: Union[Format, str] = Format.WAV,
    num_generations: int = 1,
    acting_instructions: Optional[str] = None,
) -> bytes:
    """
    Generates audio from text using Hume TTS API (synchronous version).

    Args:
        text: Text to convert to speech
        description: Description of how the voice should sound (only used if voice is None)
        voice: Name of a saved voice to use
        context_generation_id: Generation ID for contextual continuity
        format: Output audio format (wav, pcm)
        num_generations: Number of variations to generate (1-5)
        acting_instructions: Instructions for voice modulation (only used if voice is provided)

    Returns:
        Audio data as bytes
    """
    format_enum = convert_to_enum(Format, format)
    client = get_client()

    utterance_params: Dict[str, Any] = {"text": text}

    # Set voice based on parameters
    if voice:
        utterance_params["voice"] = PostedUtteranceVoiceWithName(name=voice)
        # If acting_instructions provided with voice, use as description
        if acting_instructions:
            utterance_params["description"] = acting_instructions
    elif description:
        utterance_params["description"] = description

    utterance = PostedUtterance(**utterance_params)

    # Prepare API call parameters
    api_kwargs: Dict[str, Any] = {"utterances": [utterance], "num_generations": num_generations}

    # Add context parameter if available
    if context_generation_id:
        api_kwargs["context"] = PostedContextWithGenerationId(generation_id=context_generation_id)

    # Add format parameter only for PCM format
    if format_enum == Format.PCM:
        api_kwargs["format"] = FormatPcm(type="pcm")

    # Call the API and get the response
    response = client.tts.synthesize_json(**api_kwargs)

    # Return the audio data from the first generation
    return base64.b64decode(response.generations[0].audio)


@timeout(TIMEOUT)
async def agenerate(
    text: str,
    description: Optional[str] = None,
    voice: Optional[str] = None,
    context_generation_id: Optional[str] = None,
    format: Union[Format, str] = Format.WAV,
    num_generations: int = 1,
    acting_instructions: Optional[str] = None,
) -> bytes:
    """
    Asynchronously generates audio from text using Hume TTS API.

    Args:
        text: Text to convert to speech
        description: Description of how the voice should sound (only used if voice is None)
        voice: Name of a saved voice to use
        context_generation_id: Generation ID for contextual continuity
        format: Output audio format (wav, pcm)
        num_generations: Number of variations to generate (1-5)
        acting_instructions: Instructions for voice modulation (only used if voice is provided)

    Returns:
        Audio data as bytes
    """
    format_enum = convert_to_enum(Format, format)
    client = get_aclient()

    utterance_params: Dict[str, Any] = {"text": text}

    # Set voice based on parameters
    if voice:
        utterance_params["voice"] = PostedUtteranceVoiceWithName(name=voice)
        # If acting_instructions provided with voice, use as description
        if acting_instructions:
            utterance_params["description"] = acting_instructions
    elif description:
        utterance_params["description"] = description

    utterance = PostedUtterance(**utterance_params)

    # Prepare API call parameters
    api_kwargs: Dict[str, Any] = {"utterances": [utterance], "num_generations": num_generations}

    # Add context parameter if available
    if context_generation_id:
        api_kwargs["context"] = PostedContextWithGenerationId(generation_id=context_generation_id)

    # Add format parameter only for PCM format
    if format_enum == Format.PCM:
        api_kwargs["format"] = FormatPcm(type="pcm")

    # Call API and get response
    response = await client.tts.synthesize_json(**api_kwargs)

    # Return the first generation's audio as bytes
    audio_base64 = response.generations[0].audio
    return base64.b64decode(audio_base64)
