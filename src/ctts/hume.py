import asyncio
import base64
from enum import Enum
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from hume import AsyncHumeClient
from hume.tts import FormatPcm, PostedContextWithGenerationId, PostedUtterance, PostedUtteranceVoiceWithName

from ctts.config import get_settings
from ctts.utils import convert_to_enum


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


async def agenerate_with_continuity(
    texts: List[str],
    description: Optional[str] = None,
    voice_name: Optional[str] = None,
    format: Union[Format, str] = Format.WAV,
    acting_instructions: Optional[str] = None,
) -> List[bytes]:
    """
    Generates audio for multiple texts with continuity between each generation.

    Args:
        texts: List of texts to convert to speech
        description: Description of how the voice should sound (only used if voice_name is None)
        voice_name: Name of a saved voice to use
        format: Output audio format (wav, pcm)
        acting_instructions: Instructions for voice modulation (only used if voice_name is provided)

    Returns:
        List of audio data as bytes
    """
    if not texts:
        return []

    results = []
    context_generation_id = None

    for i, text in enumerate(texts):
        # For the first text, there is no context_generation_id
        audio = await agenerate(
            text=text,
            description=description,
            voice_name=voice_name,
            context_generation_id=context_generation_id,
            format=format,
            acting_instructions=acting_instructions,
        )
        results.append(audio)

        # For each subsequent text, use the generation_id from the previous response
        if i < len(texts) - 1:
            # Need to call the API again to get the generation_id
            client = get_client()
            format_enum = convert_to_enum(Format, format)

            utterance_params: Dict[str, Any] = {"text": text}
            if voice_name:
                utterance_params["voice"] = PostedUtteranceVoiceWithName(name=voice_name)
                if acting_instructions:
                    utterance_params["description"] = acting_instructions
            elif description:
                utterance_params["description"] = description

            utterance = PostedUtterance(**utterance_params)

            # Prepare the API call parameters
            api_kwargs: Dict[str, Any] = {"utterances": [utterance]}

            # Add context parameter if available
            if context_generation_id:
                api_kwargs["context"] = PostedContextWithGenerationId(generation_id=context_generation_id)

            # Add format parameter only for PCM format
            if format_enum == Format.PCM:
                api_kwargs["format"] = FormatPcm(type="pcm")

            response = await client.tts.synthesize_json(**api_kwargs)

            context_generation_id = response.generations[0].generation_id

    return results


def generate_with_continuity(
    texts: List[str],
    description: Optional[str] = None,
    voice_name: Optional[str] = None,
    format: Union[Format, str] = Format.WAV,
    acting_instructions: Optional[str] = None,
) -> List[bytes]:
    """
    Synchronously generates audio for multiple texts with continuity between each generation.

    Args:
        texts: List of texts to convert to speech
        description: Description of how the voice should sound (only used if voice_name is None)
        voice_name: Name of a saved voice to use
        format: Output audio format (wav, pcm)
        acting_instructions: Instructions for voice modulation (only used if voice_name is provided)

    Returns:
        List of audio data as bytes
    """
    try:
        # Check if an event loop is already running
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # For Jupyter/IPython we use nest_asyncio
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(
                agenerate_with_continuity(
                    texts=texts,
                    description=description,
                    voice_name=voice_name,
                    format=format,
                    acting_instructions=acting_instructions,
                )
            )
        else:
            # For regular scripts
            return loop.run_until_complete(
                agenerate_with_continuity(
                    texts=texts,
                    description=description,
                    voice_name=voice_name,
                    format=format,
                    acting_instructions=acting_instructions,
                )
            )
    except RuntimeError:
        # If no loop is available, use asyncio.run()
        return asyncio.run(
            agenerate_with_continuity(
                texts=texts,
                description=description,
                voice_name=voice_name,
                format=format,
                acting_instructions=acting_instructions,
            )
        )


async def asave_voice(generation_id: str, name: str) -> str:
    """
    Saves a voice to the voice library for future use (asynchronous version).

    Args:
        generation_id: The generation ID of the voice to save
        name: Name to give to the saved voice

    Returns:
        The ID of the saved voice
    """
    client = get_client()
    voice = await client.tts.voices.create(name=name, generation_id=generation_id)
    if voice.id is None:
        raise ValueError("Failed to create voice, ID is None")
    return voice.id


def save_voice(generation_id: str, name: str) -> str:
    """
    Saves a voice to the voice library for future use (synchronous version).

    Args:
        generation_id: The generation ID of the voice to save
        name: Name to give to the saved voice

    Returns:
        The ID of the saved voice
    """
    try:
        # Check if an event loop is already running
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # For Jupyter/IPython we use nest_asyncio
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(asave_voice(generation_id=generation_id, name=name))
        else:
            # For regular scripts
            return loop.run_until_complete(asave_voice(generation_id=generation_id, name=name))
    except RuntimeError:
        # If no loop is available, use asyncio.run()
        return asyncio.run(asave_voice(generation_id=generation_id, name=name))


async def stream(
    texts: List[str],
    description: Optional[str] = None,
    voice_name: Optional[str] = None,
    context_generation_id: Optional[str] = None,
    format: Union[Format, str] = Format.PCM,
    acting_instructions: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    """
    Streams audio for a list of utterances using Hume TTS API.

    Args:
        texts: List of texts to convert to speech
        description: Description of how the voice should sound (only used if voice_name is None)
        voice_name: Name of a saved voice to use
        context_generation_id: Generation ID for contextual continuity
        format: Output audio format (wav, pcm)
        acting_instructions: Instructions for voice modulation (only used if voice_name is provided)

    Yields:
        Audio data chunks as bytes
    """
    if not texts:
        return

    format_enum = convert_to_enum(Format, format)
    client = get_client()

    # Prepare utterances
    utterances = []
    for text in texts:
        utterance_params: Dict[str, Any] = {"text": text}

        # Set voice based on parameters
        if voice_name:
            utterance_params["voice"] = PostedUtteranceVoiceWithName(name=voice_name)
            # If acting_instructions provided with voice_name, use as description
            if acting_instructions:
                utterance_params["description"] = acting_instructions
        elif description:
            utterance_params["description"] = description

        utterances.append(PostedUtterance(**utterance_params))

    # Prepare the API call parameters
    api_kwargs: Dict[str, Any] = {"utterances": utterances}

    # Add context parameter if available
    if context_generation_id:
        api_kwargs["context"] = PostedContextWithGenerationId(generation_id=context_generation_id)

    # Add format parameter only for PCM format
    if format_enum == Format.PCM:
        api_kwargs["format"] = FormatPcm(type="pcm")

    # Stream audio
    async for snippet in client.tts.synthesize_json_streaming(**api_kwargs):
        yield base64.b64decode(snippet.audio)
