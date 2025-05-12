import logging
from enum import Enum
from typing import cast

from timeout_function_decorator import timeout

from .base import ProviderClient
from .replicate import run
from .utils import convert_to_enum

logger = logging.getLogger(__name__)


class Voice(str, Enum):
    """Available speakers for Orpheus TTS model."""

    TARA = "tara"
    LEAH = "leah"
    JESS = "jess"
    LEO = "leo"
    DAN = "dan"
    MIA = "mia"
    ZAC = "zac"
    ZOE = "zoe"


class Model(str, Enum):
    """Available models for Orpheus TTS."""

    ORPHEUS_3B = "lucataco/orpheus-3b-0.1-ft:79f2a473e6a9720716a473d9b2f2951437dbf91dc02ccb7079fb3d89b881207f"


class EmotiveTags(str, Enum):
    LAUGH = "`<laugh>`"
    CHUCKLE = "`<chuckle>`"
    SIGH = "`<sigh>`"
    COUGH = "`<cough>`"
    SNIFFLE = "`<sniffle>`"
    GROAN = "`<groan>`"
    YAWN = "`<yawn>`"
    GASP = "`<gasp>`"


class OrpheusClient(ProviderClient):
    """Orpheus text-to-speech API client."""

    def __init__(self, api_key: str, timeout: float):
        self.api_key = api_key
        self.timeout = timeout

    def get_client(self, api_key: str) -> None:
        # Orpheus uses Replicate API and doesn't need a dedicated client
        return None

    def generate(
        self,
        text: str,
        speaker: Voice | str = Voice.TARA,
        model: Model | str = Model.ORPHEUS_3B,
        language: str = "en",
        top_p: float = 0.95,
        top_k: int = 50,
        temperature: float = 0.6,
        max_new_tokens: int = 1200,
        repetition_penalty: float = 1.1,
        duration_scale: float = 1.0,
        output_format: str = "mp3",
    ) -> bytes:
        """
        Generates audio from text using the Orpheus TTS model via Replicate.

        Args:
            text: Text to convert to speech
            speaker: Speaker type (male, female, or tara)
            model: TTS model to use
            language: Language code
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            temperature: Temperature for sampling
            max_new_tokens: Maximum number of new tokens to generate
            repetition_penalty: Penalty for repetition
            duration_scale: Speech duration scaling factor
            output_format: Output format (mp3 or wav)

        Returns:
            Audio data as bytes
        """
        logger.debug(
            "orpheus.generate(text=%s, speaker=%s, model=%s, language=%s, top_p=%s, top_k=%s, temperature=%s, max_new_tokens=%s, repetition_penalty=%s, duration_scale=%s, output_format=%s)",  # noqa: E501
            text,
            speaker,
            model,
            language,
            top_p,
            top_k,
            temperature,
            max_new_tokens,
            repetition_penalty,
            duration_scale,
            output_format,
        )
        timed_func = timeout(self.timeout)(self._generate)
        return cast(
            bytes,
            timed_func(
                text,
                speaker,
                model,
                language,
                top_p,
                top_k,
                temperature,
                max_new_tokens,
                repetition_penalty,
                duration_scale,
                output_format,
            ),
        )

    def _generate(
        self,
        text: str,
        speaker: Voice | str = Voice.TARA,
        model: Model | str = Model.ORPHEUS_3B,
        language: str = "en",
        top_p: float = 0.95,
        top_k: int = 50,
        temperature: float = 0.6,
        max_new_tokens: int = 1200,
        repetition_penalty: float = 1.1,
        duration_scale: float = 1.0,
        output_format: str = "mp3",
    ) -> bytes:
        speaker_str = convert_to_enum(Voice, speaker).value
        model_str = convert_to_enum(Model, model).value
        input_data = {
            "text": text,
            "speaker": speaker_str,
            "language": language,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "duration_scale": duration_scale,
            "output_format": output_format,
        }
        return run(self.api_key, model_str, input_data)
