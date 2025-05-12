import base64
import logging
from enum import Enum
from typing import Any, Dict, Optional, Union, cast

from hume import HumeClient
from hume.tts import FormatPcm, PostedContextWithGenerationId, PostedUtterance, PostedUtteranceVoiceWithName
from timeout_function_decorator import timeout

from .base import ProviderClient
from .utils import convert_to_enum

logger = logging.getLogger(__name__)


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


class HumeProviderClient(ProviderClient):
    """Hume text-to-speech API client."""

    def __init__(self, api_key: str, timeout: float):
        self.api_key = api_key
        self.timeout = timeout
        self.client = self.get_client(api_key)

    def get_client(self, api_key: str) -> HumeClient:
        return HumeClient(api_key=api_key)

    def generate(
        self,
        text: str,
        description: Optional[str] = None,
        voice: Optional[str] = None,
        context_generation_id: Optional[str] = None,
        format: Union[Format, str] = Format.WAV,
        num_generations: int = 1,
        acting_instructions: Optional[str] = None,
    ) -> bytes:
        """
        Generates audio from text using Hume TTS API.

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
        logger.debug(
            "hume.generate(text=%s, description=%s, voice=%s, context_generation_id=%s, format=%s, num_generations=%s, acting_instructions=%s)",  # noqa: E501
            text,
            description,
            voice,
            context_generation_id,
            format,
            num_generations,
            acting_instructions,
        )
        timed_func = timeout(self.timeout)(self._generate)
        return cast(
            bytes,
            timed_func(text, description, voice, context_generation_id, format, num_generations, acting_instructions),
        )

    def _generate(
        self,
        text: str,
        description: Optional[str] = None,
        voice: Optional[str] = None,
        context_generation_id: Optional[str] = None,
        format: Union[Format, str] = Format.WAV,
        num_generations: int = 1,
        acting_instructions: Optional[str] = None,
    ) -> bytes:
        format_enum = convert_to_enum(Format, format)

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
        response = self.client.tts.synthesize_json(**api_kwargs)

        # Return the audio data from the first generation
        return base64.b64decode(response.generations[0].audio)
