import asyncio
import base64
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional, Union

from hume import AsyncHumeClient
from hume.tts import FormatPcm, PostedContextWithGenerationId, PostedUtterance, PostedUtteranceVoiceWithName

from ctts.config import TIMEOUT, get_settings
from ctts.utils import async_timeout, convert_to_enum, timeout


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


@lru_cache()
def get_client() -> AsyncHumeClient:
    """Returns a Hume client."""
    settings = get_settings().hume
    assert settings is not None, "Hume settings are not configured"
    return AsyncHumeClient(api_key=settings.api_key)


@async_timeout(TIMEOUT)
async def agenerate(
    text: str,
    description: Optional[str] = None,
    voice_name: Optional[str] = None,
    context_generation_id: Optional[str] = None,
    format: Union[Format, str] = Format.WAV,
    num_generations: int = 1,
    acting_instructions: Optional[str] = None,
) -> bytes:
    """
    Asynchronously generates audio from text using Hume TTS API.

    Args:
        text: Text to convert to speech
        description: Description of how the voice should sound (only used if voice_name is None)
        voice_name: Name of a saved voice to use
        context_generation_id: Generation ID for contextual continuity
        format: Output audio format (wav, pcm)
        num_generations: Number of variations to generate (1-5)
        acting_instructions: Instructions for voice modulation (only used if voice_name is provided)

    Returns:
        Audio data as bytes
    """
    format_enum = convert_to_enum(Format, format)
    client = get_client()

    utterance_params: Dict[str, Any] = {"text": text}

    # Set voice based on parameters
    if voice_name:
        utterance_params["voice"] = PostedUtteranceVoiceWithName(name=voice_name)
        # If acting_instructions provided with voice_name, use as description
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


@timeout(TIMEOUT)
def generate(
    text: str,
    description: Optional[str] = None,
    voice_name: Optional[str] = None,
    context_generation_id: Optional[str] = None,
    format: Union[Format, str] = Format.WAV,
    num_generations: int = 1,
    acting_instructions: Optional[str] = None,
) -> bytes:
    """
    Generates audio from text using Hume TTS API (synchronous version).

    Args:
        text: Text to convert to speech
        description: Description of how the voice should sound (only used if voice_name is None)
        voice_name: Name of a saved voice to use
        context_generation_id: Generation ID for contextual continuity
        format: Output audio format (wav, pcm)
        num_generations: Number of variations to generate (1-5)
        acting_instructions: Instructions for voice modulation (only used if voice_name is provided)

    Returns:
        Audio data as bytes
    """
    try:
        # Check if an event loop is already running
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # For Jupyter/IPython we use nest_asyncio
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(
                agenerate(
                    text=text,
                    description=description,
                    voice_name=voice_name,
                    context_generation_id=context_generation_id,
                    format=format,
                    num_generations=num_generations,
                    acting_instructions=acting_instructions,
                )
            )
        else:
            # For regular scripts
            return loop.run_until_complete(
                agenerate(
                    text=text,
                    description=description,
                    voice_name=voice_name,
                    context_generation_id=context_generation_id,
                    format=format,
                    num_generations=num_generations,
                    acting_instructions=acting_instructions,
                )
            )
    except RuntimeError:
        # If no loop is available, use asyncio.run()
        return asyncio.run(
            agenerate(
                text=text,
                description=description,
                voice_name=voice_name,
                context_generation_id=context_generation_id,
                format=format,
                num_generations=num_generations,
                acting_instructions=acting_instructions,
            )
        )
